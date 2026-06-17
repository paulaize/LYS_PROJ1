/*
 * v1 IHC export — writes one CSV row per section/annotation.
 *
 * Headless usage after detect_cells.groovy:
 *   QuPath script --image "<path>.vsi" \
 *     --args "A,/abs/path/out.csv,1,500,8,tissue_v1,section_01,exploratory_not_final" export_measurements.groovy
 *
 * Argument order:
 *   panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample,
 *   tissue_annotation_name,section_id,threshold_status
 *
 * The channel index and threshold must come from confirmed config/animal facts.
 * Do not trust example values. FITC is exported as IgG-FITC signal, not direct
 * LYS241 concentration.
 */

import qupath.lib.regions.RegionRequest
import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample,tissue_annotation_name,section_id,threshold_status")
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
def tissueName = parts.length > 5 && parts[5] ? parts[5] : "tissue_v1"
def sectionId = parts.length > 6 && parts[6] ? parts[6] : ""
def thresholdStatus = parts.length > 7 && parts[7] ? parts[7] : "exploratory_not_final"

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

def tissue = getAnnotationObjects().find { it.getName() == tissueName }
if (tissue == null) {
    throw new IllegalStateException("Missing reviewed tissue annotation '${tissueName}'. Run detect_cells.groovy, then review/correct it in QuPath before export.")
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
int W = img.getWidth(); int H = img.getHeight()
int bands = raster.getNumBands()
if (IGG_FITC_CH >= bands) {
    throw new IllegalArgumentException("Configured IgG-FITC channel ${IGG_FITC_CH} but image has only ${bands} bands")
}

long iggFitcPosPx = 0
long totalPx = 0
for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
        double fullX = reqX + (x + 0.5) * downsample
        double fullY = reqY + (y + 0.5) * downsample
        if (!roi.contains(fullX, fullY)) continue
        totalPx++
        double v = raster.getSampleDouble(x, y, IGG_FITC_CH)
        if (v > IGG_FITC_THRESHOLD) iggFitcPosPx++
    }
}
if (totalPx == 0) {
    throw new IllegalStateException("Annotation '${tissueName}' sampled to zero pixels at downsample ${downsample}")
}

double scale = downsample * downsample * pxAreaUm2
double iggFitcPosAreaUm2 = iggFitcPosPx * scale
double totalAreaUm2 = totalPx * scale

// DAPI count placeholder. Milestone 2 replaces with StarDist/InstanSeg.
int dapiCount = -1

def name = getProjectEntry() ? getProjectEntry().getImageName() : server.getMetadata().getName()
def qcFlag = thresholdStatus == "approved" ? "v1_tissue_area_only" : "v1_tissue_area_only;threshold_${thresholdStatus}"
def header = "image,panel,section_id,region,tissue_annotation,igg_fitc_channel_index,igg_fitc_threshold,threshold_status,downsample,igg_fitc_pos_area_um2,total_area_um2,dapi_count,qc_flag\n"
def row = "${csv(name)},${csv(panel)},${csv(sectionId)},${csv(tissueName)},${csv(tissueName)},${IGG_FITC_CH},${IGG_FITC_THRESHOLD},${csv(thresholdStatus)},${downsample},${iggFitcPosAreaUm2},${totalAreaUm2},${dapiCount},${csv(qcFlag)}\n"

def f = new File(outPath)
if (!f.exists()) f.text = header
f.append(row)
print "Wrote IgG-FITC measurement row for ${name} (panel ${panel}, ${tissueName}, threshold_status=${thresholdStatus}) -> ${outPath}"

String csv(value) {
    def s = value == null ? "" : value.toString()
    return "\"" + s.replace("\"", "\"\"") + "\""
}
