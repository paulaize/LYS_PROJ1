/*
 * v1 IHC section QC thumbnail exporter.
 *
 * This is for manual section selection before threshold calibration. It writes
 * one low-resolution composite PNG for the opened Bio-Formats series and
 * appends a manifest row with blank manual QC fields.
 *
 * Argument order:
 *   panel,out_dir,manifest_csv,section_id,series_index,source_animal,
 *   source_role,max_thumbnail_px,expected_channels
 */

import qupath.lib.regions.RegionRequest
import javax.imageio.ImageIO
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.util.Arrays
import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_dir,manifest_csv,section_id,series_index,source_animal,source_role,max_thumbnail_px")
}

def parts = args[0].split(",", -1)
if (parts.length < 7) {
    throw new IllegalArgumentException("Expected args: panel,out_dir,manifest_csv,section_id,series_index,source_animal,source_role[,max_thumbnail_px,expected_channels]")
}

def panel = parts[0]
def outDir = new File(parts[1])
def manifestPath = parts[2]
def sectionId = parts[3]
int seriesIndex = parts[4] as int
def sourceAnimal = parts[5]
def sourceRole = parts[6]
int maxThumbnailPx = parts.length > 7 && parts[7] ? parts[7] as int : 1400
int expectedChannels = parts.length > 8 && parts[8] ? parts[8] as int : 0

if (maxThumbnailPx <= 0) {
    throw new IllegalArgumentException("max_thumbnail_px must be a positive integer")
}
if (expectedChannels < 0) {
    throw new IllegalArgumentException("expected_channels must be zero or a positive integer")
}
if (!outDir.exists() && !outDir.mkdirs()) {
    throw new IllegalStateException("Could not create output directory: ${outDir}")
}

def imageData = getCurrentImageData()
def server = imageData.getServer()
def metadata = server.getMetadata()
def cal = server.getPixelCalibration()
def name = getProjectEntry() ? getProjectEntry().getImageName() : metadata.getName()

int fullW = server.getWidth()
int fullH = server.getHeight()
double downsample = Math.max(1.0, Math.max(fullW, fullH) / (double)maxThumbnailPx)

def request = RegionRequest.createInstance(server.getPath(), downsample, 0, 0, fullW, fullH)
def img = server.readRegion(request)
def raster = img.getRaster()
int W = img.getWidth()
int H = img.getHeight()
int bands = raster.getNumBands()
if (bands <= 0) {
    throw new IllegalStateException("Image has no readable bands")
}

double[] mins = new double[bands]
double[] maxs = new double[bands]
for (int b = 0; b < bands; b++) {
    double[] values = new double[W * H]
    int idx = 0
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            values[idx++] = raster.getSampleDouble(x, y, b)
        }
    }
    Arrays.sort(values)
    mins[b] = percentileSorted(values, 0.01)
    maxs[b] = percentileSorted(values, 0.995)
    if (!Double.isFinite(mins[b]) || !Double.isFinite(maxs[b]) || maxs[b] <= mins[b]) {
        mins[b] = values[0]
        maxs[b] = values[values.length - 1]
    }
    if (maxs[b] <= mins[b]) {
        maxs[b] = mins[b] + 1.0
    }
}

def composite = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB)
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        int r = 0
        int g = 0
        int bl = 0
        for (int b = 0; b < bands; b++) {
            int v = toByte(raster.getSampleDouble(x, y, b), mins[b], maxs[b])
            int[] c = channelColor(b)
            r = Math.max(r, (int)Math.round(v * c[0] / 255.0))
            g = Math.max(g, (int)Math.round(v * c[1] / 255.0))
            bl = Math.max(bl, (int)Math.round(v * c[2] / 255.0))
        }
        composite.setRGB(x, y, rgb(r, g, bl))
    }
}

def thumbnailPath = new File(outDir, "section_qc_composite.png")
writePng(composite, thumbnailPath, maxThumbnailPx)

double umPerPxX = cal.getPixelWidthMicrons()
double umPerPxY = cal.getPixelHeightMicrons()
def pixelStatus = (
        Double.isFinite(umPerPxX) && Double.isFinite(umPerPxY) &&
        umPerPxX > 0 && umPerPxY > 0
) ? "ok" : "invalid_pixel_calibration"
def channelStatus = (
        expectedChannels > 0 && bands != expectedChannels
) ? "unexpected_channel_count" : "ok"
def qcFlags = [pixelStatus, channelStatus].findAll { it != "ok" }
def qcFlag = qcFlags.isEmpty() ? "ok" : qcFlags.join(";")

def header = "source_animal,source_role,panel,section_id,series_index,image,width_px,height_px,channels,expected_channels,pixel_width_um,pixel_height_um,downsample,thumbnail_path,section_qc_status,section_qc_notes,selected_for_threshold_calibration,selected_for_final_export,qc_flag\n"
def manifest = new File(manifestPath)
if (!manifest.exists()) manifest.text = header
def row = "${csv(sourceAnimal)},${csv(sourceRole)},${csv(panel)},${csv(sectionId)},${seriesIndex},${csv(name)},${fullW},${fullH},${bands},${expectedChannels},${umPerPxX},${umPerPxY},${downsample},${csv(thumbnailPath.getAbsolutePath())},\"pending_review\",\"\",\"\",\"\",${csv(qcFlag)}\n"
manifest.append(row)

print "Wrote section QC thumbnail for ${name} panel ${panel} ${sectionId} -> ${thumbnailPath}"

int[] channelColor(int band) {
    if (band == 0) return [64, 120, 255] as int[]
    if (band == 1) return [0, 255, 80] as int[]
    if (band == 2) return [255, 180, 0] as int[]
    if (band == 3) return [255, 50, 80] as int[]
    return [180, 180, 180] as int[]
}

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

double percentileSorted(double[] sorted, double p) {
    if (sorted.length == 0) return Double.NaN
    double clamped = Math.max(0.0, Math.min(1.0, p))
    int idx = (int)Math.round(clamped * (sorted.length - 1))
    return sorted[idx]
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
