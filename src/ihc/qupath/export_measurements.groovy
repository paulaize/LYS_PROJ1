/*
 * v1 IHC export — writes one CSV row per section/annotation.
 *
 * Headless usage after reviewed tissue/artifact annotations exist in a project:
 *   QuPath script --project "<project.qpproj>" --image "<project image name>" \
 *     --args "A,/abs/path/out.csv,1,500,8,tissue_v1,section_01,approved,artifact_exclude,require_reviewed,BD_08_5D,target" export_measurements.groovy
 *
 * Argument order:
 *   panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample,
 *   tissue_annotation_name,section_id,threshold_status,artifact_annotation_names,
 *   tissue_mode,source_animal,source_role
 *
 * The channel index and threshold must come from confirmed config/animal facts.
 * Do not trust example values. FITC is exported as IgG-FITC signal, not direct
 * LYS241 concentration.
 */

import qupath.lib.objects.PathObjects
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample,tissue_annotation_name,section_id,threshold_status")
}

def parts = args[0].split(",", -1)
if (parts.length < 4) {
    throw new IllegalArgumentException("Expected args: panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold[,downsample]")
}

def panel = parts[0]
def outPath = parts[1]
int IGG_FITC_CH = parts[2] as int
double IGG_FITC_THRESHOLD = parts[3] as double
double downsample = parts.length > 4 ? parts[4] as double : 8.0
def tissueName = parts.length > 5 && parts[5] ? parts[5] : "tissue_v1"
def sectionId = parts.length > 6 && parts[6] ? parts[6] : ""
def thresholdStatus = parts.length > 7 && parts[7] ? parts[7] : "exploratory_not_final"
def artifactNames = parts.length > 8 && parts[8] ? splitNames(parts[8]) : ["artifact_exclude"]
def tissueMode = parts.length > 9 && parts[9] ? parts[9] : "require_reviewed"
def sourceAnimal = parts.length > 10 && parts[10] ? parts[10] : ""
def sourceRole = parts.length > 11 && parts[11] ? parts[11] : ""

if (IGG_FITC_CH < 0) {
    throw new IllegalArgumentException("igg_fitc_channel_index must be >= 0")
}
if (IGG_FITC_THRESHOLD <= 0) {
    throw new IllegalArgumentException("igg_fitc_threshold must be > 0 and tuned before trusting outputs")
}

def imageData = getCurrentImageData()
def server = imageData.getServer()
def cal = server.getPixelCalibration()
double umPerPxX = cal.getPixelWidthMicrons()
double umPerPxY = cal.getPixelHeightMicrons()
if (!Double.isFinite(umPerPxX) || !Double.isFinite(umPerPxY) || umPerPxX <= 0 || umPerPxY <= 0) {
    throw new IllegalStateException("Missing or invalid pixel calibration; cannot compute tissue area")
}
double pxAreaUm2 = umPerPxX * umPerPxY

def tissueCreated = false
def tissue = getAnnotationObjects().find { it.getName() == tissueName }
if (tissue == null) {
    if (tissueMode == "auto_if_missing") {
        tissue = createRoughTissueAnnotation(server, tissueName, downsample)
        tissueCreated = true
        addObject(tissue)
    } else {
        throw new IllegalStateException("Missing reviewed tissue annotation '${tissueName}'. Run detect_cells.groovy, then review/correct it in QuPath before export, or use tissue_mode=auto_if_missing only for exploratory draft exports.")
    }
}
def roi = tissue.getROI()
if (roi == null) {
    throw new IllegalStateException("Annotation '${tissueName}' has no ROI")
}
def artifactRois = getAnnotationObjects()
        .findAll { artifactNames.contains(it.getName()) && it.getROI() != null }
        .collect { it.getROI() }

int reqX = Math.max(0, (int)Math.floor(roi.getBoundsX()))
int reqY = Math.max(0, (int)Math.floor(roi.getBoundsY()))
int reqW = Math.min(server.getWidth() - reqX, (int)Math.ceil(roi.getBoundsWidth()))
int reqH = Math.min(server.getHeight() - reqY, (int)Math.ceil(roi.getBoundsHeight()))
if (reqW <= 0 || reqH <= 0) {
    throw new IllegalStateException("Annotation '${tissueName}' has empty bounds")
}

def request = RegionRequest.createInstance(server.getPath(), downsample, reqX, reqY, reqW, reqH)
def img = server.readRegion(request)
def raster = img.getRaster()
int W = img.getWidth(); int H = img.getHeight()
int bands = raster.getNumBands()
if (IGG_FITC_CH >= bands) {
    throw new IllegalArgumentException("Configured IgG-FITC channel ${IGG_FITC_CH} but image has only ${bands} bands")
}

long iggFitcPosPx = 0
long totalPx = 0
long tissuePx = 0
long artifactExcludedPx = 0
double iggFitcIntensitySum = 0.0
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        double fullX = reqX + (x + 0.5) * downsample
        double fullY = reqY + (y + 0.5) * downsample
        if (!roi.contains(fullX, fullY)) continue
        tissuePx++
        if (insideAny(artifactRois, fullX, fullY)) {
            artifactExcludedPx++
            continue
        }
        totalPx++
        double v = raster.getSampleDouble(x, y, IGG_FITC_CH)
        iggFitcIntensitySum += v
        if (v > IGG_FITC_THRESHOLD) iggFitcPosPx++
    }
}
if (totalPx == 0) {
    throw new IllegalStateException("Annotation '${tissueName}' sampled to zero valid pixels after artifact exclusion at downsample ${downsample}")
}

double scale = downsample * downsample * pxAreaUm2
double iggFitcPosAreaUm2 = iggFitcPosPx * scale
double totalAreaUm2 = totalPx * scale
double tissueAreaUm2 = tissuePx * scale
double artifactExcludedAreaUm2 = artifactExcludedPx * scale
double iggFitcPctPositiveArea = iggFitcPosAreaUm2 / totalAreaUm2 * 100.0
double iggFitcMeanIntensity = iggFitcIntensitySum / totalPx
double analysisResolutionUmX = downsample * umPerPxX
double analysisResolutionUmY = downsample * umPerPxY

def name = getProjectEntry() ? getProjectEntry().getImageName() : server.getMetadata().getName()
def tissueQc = tissueCreated ? "rough_tissue_auto_unreviewed" : (tissueMode == "require_reviewed" ? "tissue_annotation_present_review_required" : "tissue_annotation_present")
def thresholdFlag = thresholdStatus.startsWith("threshold_") ? thresholdStatus : "threshold_${thresholdStatus}"
def thresholdQc = thresholdStatus == "approved" ? "" : ";${thresholdFlag}"
def artifactQc = artifactRois.isEmpty() ? ";no_artifact_exclusion_annotations" : ";artifact_excluded"
def tissueQcFlag = tissueCreated ? ";rough_tissue_auto_unreviewed" : ";${tissueQc}"
def qcFlag = "v1_tissue_area_only${thresholdQc}${artifactQc}${tissueQcFlag}"
def header = "source_animal,source_role,image,panel,section_id,region,tissue_annotation,tissue_qc,artifact_annotation_names,artifact_annotation_count,igg_fitc_channel_index,igg_fitc_threshold,threshold_status,downsample,pixel_width_um,pixel_height_um,analysis_resolution_um_x,analysis_resolution_um_y,igg_fitc_pos_area_um2,total_area_um2,valid_analyzed_area_um2,tissue_area_um2,artifact_excluded_area_um2,igg_fitc_pct_positive_area,igg_fitc_mean_intensity,qc_flag\n"
def row = "${csv(sourceAnimal)},${csv(sourceRole)},${csv(name)},${csv(panel)},${csv(sectionId)},${csv(tissueName)},${csv(tissueName)},${csv(tissueQc)},${csv(artifactNames.join(';'))},${artifactRois.size()},${IGG_FITC_CH},${IGG_FITC_THRESHOLD},${csv(thresholdStatus)},${downsample},${umPerPxX},${umPerPxY},${analysisResolutionUmX},${analysisResolutionUmY},${iggFitcPosAreaUm2},${totalAreaUm2},${totalAreaUm2},${tissueAreaUm2},${artifactExcludedAreaUm2},${iggFitcPctPositiveArea},${iggFitcMeanIntensity},${csv(qcFlag)}\n"

def f = new File(outPath)
if (!f.exists()) f.text = header
f.append(row)
print "Wrote IgG-FITC measurement row for ${name} (panel ${panel}, ${tissueName}, threshold_status=${thresholdStatus}) -> ${outPath}"

List<String> splitNames(String value) {
    return value.split(";").collect { it.trim() }.findAll { it }
}

boolean insideAny(rois, double x, double y) {
    for (def r : rois) {
        if (r.contains(x, y)) return true
    }
    return false
}

def createRoughTissueAnnotation(server, tissueName, downsample) {
    def request = RegionRequest.createInstance(server.getPath(), downsample,
            0, 0, server.getWidth(), server.getHeight())
    def img = server.readRegion(request)
    def raster = img.getRaster()
    int W = img.getWidth()
    int H = img.getHeight()
    int bands = raster.getNumBands()

    double[] signal = new double[W * H]
    int idx = 0
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double maxValue = 0
            for (int b = 0; b < bands; b++) {
                maxValue = Math.max(maxValue, raster.getSampleDouble(x, y, b))
            }
            signal[idx++] = maxValue
        }
    }

    double threshold = otsu(signal)
    int minX = W
    int minY = H
    int maxX = -1
    int maxY = -1
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (signal[y * W + x] > threshold) {
                minX = Math.min(minX, x)
                minY = Math.min(minY, y)
                maxX = Math.max(maxX, x)
                maxY = Math.max(maxY, y)
            }
        }
    }

    if (maxX < 0 || maxY < 0) {
        throw new IllegalStateException("Could not create ${tissueName}: no tissue-like signal found at downsample ${downsample}")
    }

    int marginPx = 200
    int fullMinX = Math.max(0, (int)Math.floor(minX * downsample) - marginPx)
    int fullMinY = Math.max(0, (int)Math.floor(minY * downsample) - marginPx)
    int fullMaxX = Math.min(server.getWidth(), (int)Math.ceil((maxX + 1) * downsample) + marginPx)
    int fullMaxY = Math.min(server.getHeight(), (int)Math.ceil((maxY + 1) * downsample) + marginPx)

    def plane = ImagePlane.getDefaultPlane()
    def roi = ROIs.createRectangleROI(fullMinX, fullMinY, fullMaxX - fullMinX, fullMaxY - fullMinY, plane)
    def annotation = PathObjects.createAnnotationObject(roi)
    annotation.setName(tissueName)
    return annotation
}

double otsu(double[] values) {
    double minValue = Double.POSITIVE_INFINITY
    double maxValue = Double.NEGATIVE_INFINITY
    for (double v : values) {
        if (v < minValue) minValue = v
        if (v > maxValue) maxValue = v
    }
    if (!Double.isFinite(minValue) || !Double.isFinite(maxValue) || maxValue <= minValue) {
        return maxValue
    }

    int bins = 256
    long[] hist = new long[bins]
    double scale = (bins - 1) / (maxValue - minValue)
    for (double v : values) {
        int b = (int)Math.floor((v - minValue) * scale)
        b = Math.max(0, Math.min(bins - 1, b))
        hist[b]++
    }

    long total = values.length
    double sum = 0
    for (int i = 0; i < bins; i++) sum += i * hist[i]

    long wB = 0
    double sumB = 0
    double bestVar = -1
    int best = 0
    for (int i = 0; i < bins; i++) {
        wB += hist[i]
        if (wB == 0) continue
        long wF = total - wB
        if (wF == 0) break
        sumB += i * hist[i]
        double mB = sumB / wB
        double mF = (sum - sumB) / wF
        double between = wB * wF * Math.pow(mB - mF, 2)
        if (between > bestVar) {
            bestVar = between
            best = i
        }
    }
    return minValue + (best / (double)(bins - 1)) * (maxValue - minValue)
}

String csv(value) {
    def s = value == null ? "" : value.toString()
    return "\"" + s.replace("\"", "\"\"") + "\""
}
