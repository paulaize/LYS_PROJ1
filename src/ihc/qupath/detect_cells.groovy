/*
 * v1 IHC setup — create a rough tissue_v1 annotation.
 *
 * Headless usage:
 *   QuPath script --image "<path>.vsi" \
 *     --args "A,tissue_v1,16,200" src/ihc/qupath/detect_cells.groovy
 *
 * v1 deliberately does NOT do cell-level detection or atlas regions. This
 * creates a rough signal-derived tissue annotation that Paul can review/correct
 * before export_measurements.groovy is trusted.
 */

import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.RegionRequest
import static qupath.lib.gui.scripting.QPEx.*

def parts = args ? args[0].split(",") : [] as String[]
def panel = parts.length > 0 && parts[0] ? parts[0] : "UNKNOWN"
def tissueName = parts.length > 1 && parts[1] ? parts[1] : "tissue_v1"
double downsample = parts.length > 2 && parts[2] ? parts[2] as double : 16.0
int marginPx = parts.length > 3 && parts[3] ? parts[3] as int : 200

setImageType('FLUORESCENCE')

def imageData = getCurrentImageData()
def server = imageData.getServer()

int w = server.getWidth()
int h = server.getHeight()
def request = RegionRequest.createInstance(server.getPath(), downsample, 0, 0, w, h)
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

int fullMinX = Math.max(0, (int)Math.floor(minX * downsample) - marginPx)
int fullMinY = Math.max(0, (int)Math.floor(minY * downsample) - marginPx)
int fullMaxX = Math.min(w, (int)Math.ceil((maxX + 1) * downsample) + marginPx)
int fullMaxY = Math.min(h, (int)Math.ceil((maxY + 1) * downsample) + marginPx)

def plane = ImagePlane.getDefaultPlane()
def roi = ROIs.createRectangleROI(fullMinX, fullMinY, fullMaxX - fullMinX, fullMaxY - fullMinY, plane)
def annotation = PathObjects.createAnnotationObject(roi)
annotation.setName(tissueName)
clearAllObjects()
addObject(annotation)

print "Panel ${panel}: rough ${tissueName} annotation created from signal bounding box."
print "Bounds: x=${fullMinX}, y=${fullMinY}, w=${fullMaxX - fullMinX}, h=${fullMaxY - fullMinY}; otsu=${threshold}; downsample=${downsample}"
print "Review/correct ${tissueName} in QuPath before trusting exported measurements."

fireHierarchyUpdate()

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
