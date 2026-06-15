/*
 * v1 IHC export — writes one CSV row per section/annotation.
 *
 * Headless usage after detect_cells.groovy:
 *   QuPath script --image "<path>.vsi" \
 *     --args "A,/abs/path/out.csv,3,500,8" export_measurements.groovy
 *
 * Argument order:
 *   panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample
 *
 * The channel index and threshold must come from confirmed config/animal facts.
 * Do not trust example values. FITC is exported as IgG-FITC signal, not direct
 * LYS241 concentration.
 */

import qupath.lib.regions.RegionRequest
import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample")
}

def parts = args[0].split(",")
if (parts.length < 4) {
    throw new IllegalArgumentException("Expected args: panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold[,downsample]")
}

def panel = parts[0]
def outPath = parts[1]
int IGG_FITC_CH = parts[2] as int
double IGG_FITC_THRESHOLD = parts[3] as double
double downsample = parts.length > 4 ? parts[4] as double : 8.0

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
double pxAreaUm2 = umPerPxX * umPerPxY

// Crude whole-section IgG-FITC positive area at a downsampled level.
def request = RegionRequest.createInstance(server.getPath(), downsample,
        0, 0, server.getWidth(), server.getHeight())
def img = server.readRegion(request)
def raster = img.getRaster()
int W = img.getWidth(); int H = img.getHeight()
int bands = raster.getNumBands()
if (IGG_FITC_CH >= bands) {
    throw new IllegalArgumentException("Configured IgG-FITC channel ${IGG_FITC_CH} but image has only ${bands} bands")
}

long iggFitcPosPx = 0
long totalPx = (long) W * H
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        double v = raster.getSampleDouble(x, y, IGG_FITC_CH)
        if (v > IGG_FITC_THRESHOLD) iggFitcPosPx++
    }
}

double scale = downsample * downsample * pxAreaUm2
double iggFitcPosAreaUm2 = iggFitcPosPx * scale
double totalAreaUm2 = totalPx * scale

// DAPI count placeholder. Milestone 2 replaces with StarDist/InstanSeg.
int dapiCount = -1

def name = getProjectEntry() ? getProjectEntry().getImageName() : server.getMetadata().getName()
def header = "image,panel,region,igg_fitc_pos_area_um2,total_area_um2,dapi_count,qc_flag\n"
def row = "\"${name}\",${panel},whole_section,${iggFitcPosAreaUm2},${totalAreaUm2},${dapiCount},v1_area_only\n"

def f = new File(outPath)
if (!f.exists()) f.text = header
f.append(row)
print "Wrote IgG-FITC measurement row for ${name} (panel ${panel}) -> ${outPath}"
