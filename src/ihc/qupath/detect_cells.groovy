/*
 * v1 IHC setup — create a whole-section annotation.
 *
 * Headless usage:
 *   QuPath script --image "<path>.vsi" --args "A" src/ihc/qupath/detect_cells.groovy
 *
 * v1 deliberately does NOT do cell-level detection or atlas regions. This
 * simply creates the whole_section annotation used by export_measurements.groovy.
 */

import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import static qupath.lib.gui.scripting.QPEx.*

def panel = args ? args[0] : "UNKNOWN"

setImageType('FLUORESCENCE')

def imageData = getCurrentImageData()
def server = imageData.getServer()

int w = server.getWidth()
int h = server.getHeight()
def plane = ImagePlane.getDefaultPlane()
def roi = ROIs.createRectangleROI(0, 0, w, h, plane)
def annotation = PathObjects.createAnnotationObject(roi)
annotation.setName("whole_section")
clearAllObjects()
addObject(annotation)

print "Panel ${panel}: whole-section annotation created (${w} x ${h} px)."
print "No channel order or thresholds are assumed here. Pass confirmed values to export_measurements.groovy."

fireHierarchyUpdate()
