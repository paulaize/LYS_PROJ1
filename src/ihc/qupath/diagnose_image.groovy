/*
 * v1 IHC QuPath/Bio-Formats diagnostic.
 *
 * This intentionally does not read image pixels. It only verifies that QuPath
 * can open the configured VSI series and records server metadata so slow
 * Bio-Formats startup can be separated from expensive threshold-sweep reads.
 *
 * Argument order:
 *   panel,out_csv,igg_fitc_channel_index,section_id,source_animal,source_role
 */

import static qupath.lib.gui.scripting.QPEx.*

if (!args || args.size() == 0) {
    throw new IllegalArgumentException("Missing args: panel,out_csv,igg_fitc_channel_index,section_id,source_animal,source_role")
}

def parts = args[0].split(",", -1)
if (parts.length < 3) {
    throw new IllegalArgumentException("Expected args: panel,out_csv,igg_fitc_channel_index[,section_id,source_animal,source_role]")
}

def panel = parts[0]
def outPath = parts[1]
int iggFitcChannel = parts[2] as int
def sectionId = parts.length > 3 ? parts[3] : ""
def sourceAnimal = parts.length > 4 ? parts[4] : ""
def sourceRole = parts.length > 5 ? parts[5] : ""

def imageData = getCurrentImageData()
def server = imageData.getServer()
def metadata = server.getMetadata()
def cal = server.getPixelCalibration()
def name = getProjectEntry() ? getProjectEntry().getImageName() : metadata.getName()

int width = server.getWidth()
int height = server.getHeight()
int channels = readSafely("channels") { server.nChannels() } as int
int zSlices = readSafely("z_slices") { server.nZSlices() } as int
int timepoints = readSafely("timepoints") { server.nTimepoints() } as int
int resolutions = readSafely("resolutions") { server.nResolutions() } as int
def downsamples = readSafely("preferred_downsamples") { server.getPreferredDownsamples().join(";") }
double umPerPxX = cal.getPixelWidthMicrons()
double umPerPxY = cal.getPixelHeightMicrons()

def channelStatus = iggFitcChannel >= 0 && iggFitcChannel < channels ? "ok" : "configured_channel_out_of_range"
def header = "source_animal,source_role,image,panel,section_id,width_px,height_px,channels,z_slices,timepoints,resolutions,preferred_downsamples,pixel_width_um,pixel_height_um,igg_fitc_channel_index,channel_status\n"
def f = new File(outPath)
if (!f.exists()) f.text = header
def row = "${csv(sourceAnimal)},${csv(sourceRole)},${csv(name)},${csv(panel)},${csv(sectionId)},${width},${height},${channels},${zSlices},${timepoints},${resolutions},${csv(downsamples)},${umPerPxX},${umPerPxY},${iggFitcChannel},${csv(channelStatus)}\n"
f.append(row)

print "Opened ${name} panel ${panel} ${sectionId}: ${width}x${height}, channels=${channels}, pixel=${umPerPxX}x${umPerPxY} um -> ${outPath}"

def readSafely(label, Closure valueReader) {
    try {
        return valueReader.call()
    } catch (Throwable t) {
        throw new IllegalStateException("Could not read QuPath server metadata '${label}': ${t.getMessage()}", t)
    }
}

String csv(value) {
    def s = value == null ? "" : value.toString()
    return "\"" + s.replace("\"", "\"\"") + "\""
}
