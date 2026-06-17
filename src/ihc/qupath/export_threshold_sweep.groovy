/*
 * v1 IHC threshold sweep — exploratory, not final quantification.
 *
 * Headless usage after tissue_v1 exists, or with auto_if_missing for
 * exploratory rough-tissue sweeps:
 *   QuPath script --image "<path>.vsi" \
 *     --args "A,/abs/path/threshold_sweep.csv,1,100;250;500;1000;2000,8,tissue_v1,section_01,auto_if_missing,BD_08_5D,target" \
 *     export_threshold_sweep.groovy
 *
 * Argument order:
 *   panel,out_csv,igg_fitc_channel_index,thresholds_semicolon_list,downsample,
 *   tissue_annotation_name,section_id,tissue_mode,source_animal,source_role
 */

import qupath.lib.objects.PathObjects
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_csv,igg_fitc_channel_index,thresholds,downsample,tissue_annotation_name,section_id")
}

def parts = args[0].split(",")
if (parts.length < 4) {
    throw new IllegalArgumentException("Expected args: panel,out_csv,igg_fitc_channel_index,thresholds[,downsample,tissue_annotation_name,section_id]")
}

def panel = parts[0]
def outPath = parts[1]
int IGG_FITC_CH = parts[2] as int
def thresholds = parts[3].split(";").findAll { it }.collect { it as double }
double downsample = parts.length > 4 ? parts[4] as double : 8.0
def tissueName = parts.length > 5 && parts[5] ? parts[5] : "tissue_v1"
def sectionId = parts.length > 6 && parts[6] ? parts[6] : ""
def tissueMode = parts.length > 7 && parts[7] ? parts[7] : "require_reviewed"
def sourceAnimal = parts.length > 8 && parts[8] ? parts[8] : ""
def sourceRole = parts.length > 9 && parts[9] ? parts[9] : ""

if (thresholds.isEmpty()) {
    throw new IllegalArgumentException("At least one threshold is required for the sweep")
}
if (IGG_FITC_CH < 0) {
    throw new IllegalArgumentException("igg_fitc_channel_index must be >= 0")
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
        throw new IllegalStateException("Missing tissue annotation '${tissueName}'. Run detect_cells.groovy, then review/correct it in QuPath before sweep.")
    }
}
def roi = tissue.getROI()
if (roi == null) {
    throw new IllegalStateException("Annotation '${tissueName}' has no ROI")
}

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
int W = img.getWidth()
int H = img.getHeight()
int bands = raster.getNumBands()
if (IGG_FITC_CH >= bands) {
    throw new IllegalArgumentException("Configured IgG-FITC channel ${IGG_FITC_CH} but image has only ${bands} bands")
}

long totalPx = 0
long[] positive = new long[thresholds.size()]
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        double fullX = reqX + (x + 0.5) * downsample
        double fullY = reqY + (y + 0.5) * downsample
        if (!roi.contains(fullX, fullY)) continue
        totalPx++
        double v = raster.getSampleDouble(x, y, IGG_FITC_CH)
        for (int i = 0; i < thresholds.size(); i++) {
            if (v > thresholds[i]) positive[i]++
        }
    }
}
if (totalPx == 0) {
    throw new IllegalStateException("Annotation '${tissueName}' sampled to zero pixels at downsample ${downsample}")
}

double scale = downsample * downsample * pxAreaUm2
double totalAreaUm2 = totalPx * scale
def name = getProjectEntry() ? getProjectEntry().getImageName() : server.getMetadata().getName()

def tissueQc = tissueCreated ? "rough_tissue_auto_unreviewed" : "tissue_annotation_present"
def header = "source_animal,source_role,image,panel,section_id,region,tissue_annotation,tissue_qc,igg_fitc_channel_index,threshold,downsample,igg_fitc_pos_area_um2,total_area_um2,igg_fitc_pct_positive_area,qc_flag\n"
def f = new File(outPath)
if (!f.exists()) f.text = header
for (int i = 0; i < thresholds.size(); i++) {
    double posArea = positive[i] * scale
    double pct = posArea / totalAreaUm2 * 100.0
    def qc = tissueCreated ? "threshold_sweep_not_final;rough_tissue_auto_unreviewed" : "threshold_sweep_not_final"
    def row = "${csv(sourceAnimal)},${csv(sourceRole)},${csv(name)},${csv(panel)},${csv(sectionId)},${csv(tissueName)},${csv(tissueName)},${csv(tissueQc)},${IGG_FITC_CH},${thresholds[i]},${downsample},${posArea},${totalAreaUm2},${pct},${csv(qc)}\n"
    f.append(row)
}

print "Wrote IgG-FITC threshold sweep for ${name} (${thresholds.size()} thresholds, panel ${panel}) -> ${outPath}"

String csv(value) {
    def s = value == null ? "" : value.toString()
    return "\"" + s.replace("\"", "\"\"") + "\""
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
