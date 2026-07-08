/*
 * v1 IHC threshold review thumbnails - exploratory, not final quantification.
 *
 * Headless usage after tissue_v1 exists, or with auto_if_missing for rough
 * exploratory review thumbnails:
 *   QuPath script --image "<path>.vsi" \
 *     --args "A,/abs/path/review_dir,1,100;250;500;1000,32,tissue_v1,section_01,auto_if_missing,BD_08_5D,target,1400" \
 *     export_threshold_review_thumbnails.groovy
 *
 * Argument order:
 *   panel,out_dir,igg_fitc_channel_index,thresholds_semicolon_list,downsample,
 *   tissue_annotation_name,section_id,tissue_mode,source_animal,source_role,
 *   max_thumbnail_px,artifact_annotation_names
 */

import qupath.lib.objects.PathObjects
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import javax.imageio.ImageIO
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.util.Arrays
import java.util.Locale
import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_dir,igg_fitc_channel_index,thresholds,downsample,tissue_annotation_name,section_id")
}

def parts = args[0].split(",", -1)
if (parts.length < 4) {
    throw new IllegalArgumentException("Expected args: panel,out_dir,igg_fitc_channel_index,thresholds[,downsample,tissue_annotation_name,section_id]")
}

def panel = parts[0]
def outDir = new File(parts[1])
int IGG_FITC_CH = parts[2] as int
def thresholds = parts[3].split(";").findAll { it }.collect { it as double }
double downsample = parts.length > 4 && parts[4] ? parts[4] as double : 8.0
def tissueName = parts.length > 5 && parts[5] ? parts[5] : "tissue_v1"
def sectionId = parts.length > 6 && parts[6] ? parts[6] : ""
def tissueMode = parts.length > 7 && parts[7] ? parts[7] : "require_reviewed"
def sourceAnimal = parts.length > 8 && parts[8] ? parts[8] : ""
def sourceRole = parts.length > 9 && parts[9] ? parts[9] : ""
int maxThumbnailPx = parts.length > 10 && parts[10] ? parts[10] as int : 1400
def artifactNames = parts.length > 11 && parts[11] ? splitNames(parts[11]) : ["artifact_exclude"]

if (thresholds.isEmpty()) {
    throw new IllegalArgumentException("At least one threshold is required for review thumbnails")
}
if (IGG_FITC_CH < 0) {
    throw new IllegalArgumentException("igg_fitc_channel_index must be >= 0")
}
if (maxThumbnailPx <= 0) {
    throw new IllegalArgumentException("max_thumbnail_px must be a positive integer")
}
if (!outDir.exists() && !outDir.mkdirs()) {
    throw new IllegalStateException("Could not create output directory: ${outDir}")
}

def imageData = getCurrentImageData()
def server = imageData.getServer()

def tissueCreated = false
def tissue = getAnnotationObjects().find { it.getName() == tissueName }
if (tissue == null) {
    if (tissueMode == "auto_if_missing") {
        tissue = createRoughTissueAnnotation(server, tissueName, downsample)
        tissueCreated = true
        addObject(tissue)
    } else {
        throw new IllegalStateException("Missing tissue annotation '${tissueName}'. Run detect_cells.groovy, then review/correct it in QuPath before exporting review thumbnails.")
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
int W = img.getWidth()
int H = img.getHeight()
int bands = raster.getNumBands()
if (IGG_FITC_CH >= bands) {
    throw new IllegalArgumentException("Configured IgG-FITC channel ${IGG_FITC_CH} but image has only ${bands} bands")
}

boolean[] inside = new boolean[W * H]
boolean[] artifact = new boolean[W * H]
double[] samples = new double[W * H]
double[] values = new double[W * H]
int totalPx = 0
int tissuePx = 0
int artifactExcludedPx = 0
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        int idx = y * W + x
        double fullX = reqX + (x + 0.5) * downsample
        double fullY = reqY + (y + 0.5) * downsample
        if (!roi.contains(fullX, fullY)) continue
        tissuePx++
        if (insideAny(artifactRois, fullX, fullY)) {
            artifact[idx] = true
            artifactExcludedPx++
            continue
        }
        inside[idx] = true
        double v = raster.getSampleDouble(x, y, IGG_FITC_CH)
        samples[idx] = v
        values[totalPx++] = v
    }
}
if (totalPx == 0) {
    throw new IllegalStateException("Annotation '${tissueName}' sampled to zero valid pixels after artifact exclusion at downsample ${downsample}")
}

double[] fitcValues = Arrays.copyOf(values, totalPx)
Arrays.sort(fitcValues)
double displayMin = percentileSorted(fitcValues, 0.01)
double displayMax = percentileSorted(fitcValues, 0.995)
if (!Double.isFinite(displayMin) || !Double.isFinite(displayMax) || displayMax <= displayMin) {
    displayMin = fitcValues[0]
    displayMax = fitcValues[fitcValues.length - 1]
}
if (displayMax <= displayMin) {
    displayMax = displayMin + 1.0
}

def rawImage = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB)
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        int idx = y * W + x
        if (artifact[idx]) {
            rawImage.setRGB(x, y, rgb(35, 20, 100))
            continue
        }
        if (!inside[idx]) {
            rawImage.setRGB(x, y, rgb(10, 10, 10))
            continue
        }
        int gray = toByte(samples[idx], displayMin, displayMax)
        rawImage.setRGB(x, y, rgb(gray, gray, gray))
    }
}

def rawPath = new File(outDir, "fitc_raw.png")
writePng(rawImage, rawPath, maxThumbnailPx)

def name = getProjectEntry() ? getProjectEntry().getImageName() : server.getMetadata().getName()
def tissueQc = tissueCreated ? "rough_tissue_auto_unreviewed" : "tissue_annotation_present"
def artifactQc = artifactRois.isEmpty() ? "no_artifact_exclusion_annotations" : "artifact_excluded"
def qc = tissueCreated ? "threshold_review_not_final;rough_tissue_auto_unreviewed;${artifactQc}" : "threshold_review_not_final;${artifactQc}"
def manifest = new File(outDir, "review_manifest.csv")
manifest.text = "source_animal,source_role,image,panel,section_id,region,tissue_annotation,tissue_qc,artifact_annotation_names,artifact_annotation_count,igg_fitc_channel_index,threshold,downsample,total_sampled_px,tissue_sampled_px,artifact_excluded_px,positive_px,igg_fitc_pct_positive_area,thumbnail_path,raw_thumbnail_path,qc_flag\n"

for (double threshold : thresholds) {
    def overlay = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB)
    long positivePx = 0
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x
            if (artifact[idx]) {
                overlay.setRGB(x, y, rgb(35, 20, 100))
                continue
            }
            if (!inside[idx]) {
                overlay.setRGB(x, y, rgb(10, 10, 10))
                continue
            }
            int gray = toByte(samples[idx], displayMin, displayMax)
            if (samples[idx] > threshold) {
                positivePx++
                overlay.setRGB(x, y, rgb(
                        blend(gray, 255, 0.70),
                        blend(gray, 48, 0.70),
                        blend(gray, 32, 0.70)
                ))
            } else {
                overlay.setRGB(x, y, rgb(gray, gray, gray))
            }
        }
    }
    def outPath = new File(outDir, "threshold_${thresholdLabel(threshold)}.png")
    writePng(overlay, outPath, maxThumbnailPx)
    double pct = positivePx / (double)totalPx * 100.0
    def row = "${csv(sourceAnimal)},${csv(sourceRole)},${csv(name)},${csv(panel)},${csv(sectionId)},${csv(tissueName)},${csv(tissueName)},${csv(tissueQc)},${csv(artifactNames.join(';'))},${artifactRois.size()},${IGG_FITC_CH},${threshold},${downsample},${totalPx},${tissuePx},${artifactExcludedPx},${positivePx},${pct},${csv(outPath.getAbsolutePath())},${csv(rawPath.getAbsolutePath())},${csv(qc)}\n"
    manifest.append(row)
}

print "Wrote IgG-FITC threshold review thumbnails for ${name} (${thresholds.size()} thresholds, panel ${panel}) -> ${outDir}"

int rgb(int r, int g, int b) {
    return ((clampByte(r) & 0xff) << 16) | ((clampByte(g) & 0xff) << 8) | (clampByte(b) & 0xff)
}

int clampByte(int value) {
    return Math.max(0, Math.min(255, value))
}

int toByte(double value, double displayMin, double displayMax) {
    double scaled = (value - displayMin) / (displayMax - displayMin)
    return clampByte((int)Math.round(scaled * 255.0))
}

int blend(int base, int overlay, double alpha) {
    return clampByte((int)Math.round(base * (1.0 - alpha) + overlay * alpha))
}

double percentileSorted(double[] sorted, double p) {
    if (sorted.length == 0) return Double.NaN
    double clamped = Math.max(0.0, Math.min(1.0, p))
    int idx = (int)Math.round(clamped * (sorted.length - 1))
    return sorted[idx]
}

String thresholdLabel(double threshold) {
    String label
    if (threshold == Math.rint(threshold)) {
        label = String.format(Locale.US, "%d", (long)threshold)
    } else {
        label = String.format(Locale.US, "%.3f", threshold)
    }
    return label.replaceAll("[^0-9A-Za-z._-]", "_")
}

void writePng(BufferedImage image, File outPath, int maxThumbnailPx) {
    int maxDim = Math.max(image.getWidth(), image.getHeight())
    if (maxDim <= maxThumbnailPx) {
        ImageIO.write(image, "PNG", outPath)
        return
    }
    double scale = maxThumbnailPx / (double)maxDim
    int newW = Math.max(1, (int)Math.round(image.getWidth() * scale))
    int newH = Math.max(1, (int)Math.round(image.getHeight() * scale))
    def resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB)
    def g2 = resized.createGraphics()
    g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g2.drawImage(image, 0, 0, newW, newH, null)
    g2.dispose()
    ImageIO.write(resized, "PNG", outPath)
}

String csv(value) {
    def s = value == null ? "" : value.toString()
    return "\"" + s.replace("\"", "\"\"") + "\""
}

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
