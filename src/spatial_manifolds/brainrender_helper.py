import numpy as np 
import numpy as np
import scipy.io
import pandas as pd
import math
from brainrender.actors import Cylinder
from brainrender import Scene
from brainrender import settings
from brainrender.actors import Points
from tifffile import imread
from collections import Counter

reference_set = imread('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/reference.tiff')
annotations_set = imread('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/annotation.tiff')
structure_set = pd.read_csv('/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2/structures.csv')

#for i in range(len(structure_set)):
#    print(str(structure_set.iloc[i]['acronym']) + " " + str(structure_set.iloc[i]['name']))

class Cylinder2(Cylinder):
    def __init__(self, pos_from, pos_to, root, color='powderblue', alpha=1, radius=350):
        from vedo import shapes
        from brainrender.actor import Actor
        mesh = shapes.Cylinder(pos=[pos_from, pos_to], c=color, r=radius, alpha=alpha)
        Actor.__init__(self, mesh, name="Cylinder", br_class="Cylinder")

# Function to convert stereotaxic coordinates to ABA CCF
# SC is an array with stereotaxic coordinates to be transformed
# Returns an array containing corresponding CCF coordinates in μm
# Conversion is from this post, which explains the opposite transformation: https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858/3
# Warning: this is very approximate
# Warning: the X, Y, Z schematic at the top of the linked post is incorrect, scroll down for correct one.


def StereoToCCF(SC = np.array([1,1,1]), angle = -0.0873):
    # Stretch
    stretch = SC/np.array([1,0.9434,1])
    # Rotate
    rotate = np.array([(stretch[0] * math.cos(angle) - stretch[1] * math.sin(angle)),
                       (stretch[0] * math.sin(angle) + stretch[1] * math.cos(angle)),
                       stretch[2]])
    #Translate
    trans = rotate + np.array([5400, 440, 5700])
    return(trans)


def CCFToStereo(CCF = np.array([1,1,1]), angle = 0.0873):
    #Translate
    trans = CCF - np.array([5400, 440, 5700])
    # Rotate
    rotate = np.array([(trans[0] * math.cos(angle) - trans[1] * math.sin(angle)),
                       (trans[0] * math.sin(angle) + trans[1] * math.cos(angle)),
                       trans[2]])
    # Stretch
    stretch = rotate*np.array([1,0.9434,1])
    return(stretch)


def read_probe_mat(probe_locs_path):
    mat = scipy.io.loadmat(probe_locs_path)
    probe_locs = np.array(mat['probe_locs'])
    return probe_locs
 

def read_borders_table(border_tables_path):
    mat = scipy.io.loadmat(border_tables_path)
    borders_table = pd.DataFrame(mat['borders_table'])
    return borders_table

def adjust_probe_locs(probe_locs):

    adjusted_probe_locs = np.array(
        [[probe_locs[0,0], probe_locs[0,1]], 
        [probe_locs[2,0], probe_locs[2,1]],
        [probe_locs[1,0], probe_locs[1,1]]]
    )*10
    return adjusted_probe_locs
    

def adjust_to_shank_offsets(probe_locs_list_SC, shank_offsets):
    # assumes shank offsets are a df with columns shank and y offset
    # y offset is the displacement of the shank along the directional vector

    corrected_probe_locs_list_SC = np.zeros_like(probe_locs_list_SC)
    corrected_probe_locs_list_CCF = np.zeros_like(probe_locs_list_SC)

    for i, probe_locs_SC in enumerate(probe_locs_list_SC):
        shank_offset = shank_offsets['y_offset'].iloc[i]
        probe_locs_CCF = probe_locs_SC.copy()

        z1, z2 = probe_locs_SC[0]
        y1, y2 = probe_locs_SC[1]
        x1, x2 = probe_locs_SC[2]

        # Calculate the direction vector
        direction_vector = np.array([z2 - z1, 
                                     y2 - y1, 
                                     x2 - x1])

        # Normalize the direction vector
        unit_vector = direction_vector / np.linalg.norm(direction_vector)

        # Scale the unit vector by the distance probe_offset in um
        scaled_vector = unit_vector * shank_offset
        
        # Calculate the new coordinates
        probe_locs_SC[0,1] = z2 + scaled_vector[0]
        probe_locs_SC[1,1] = y2 + scaled_vector[1]
        probe_locs_SC[2,1] = x2 + scaled_vector[2]

        probe_locs_CCF[:,0] = StereoToCCF(probe_locs_SC[:,0])
        probe_locs_CCF[:,1] = StereoToCCF(probe_locs_SC[:,1])

        corrected_probe_locs_list_SC[i, :, :] = probe_locs_SC
        corrected_probe_locs_list_CCF[i, :, :] = probe_locs_CCF
    
    return corrected_probe_locs_list_SC, corrected_probe_locs_list_CCF




def correct_for_left_side(probe_locs_list_CCF, do_correction=True):
    corrected_probe_locs_list_SC = np.zeros_like(probe_locs_list_CCF)
    corrected_probe_locs_list_CCF = np.zeros_like(probe_locs_list_CCF)

    for i, probe_locs_CCF in enumerate(probe_locs_list_CCF):
        probe_locs_SC = probe_locs_CCF.copy()
        probe_locs_SC[:,0] = CCFToStereo(probe_locs_CCF[:,0])
        probe_locs_SC[:,1] = CCFToStereo(probe_locs_CCF[:,1])

        if (probe_locs_SC[:,0][2] > 0) and do_correction:
            probe_locs_SC[:,0][2]*=-1
            probe_locs_SC[:,1][2]*=-1
            probe_locs_CCF[:,0] = StereoToCCF(probe_locs_SC[:,0])
            probe_locs_CCF[:,1] = StereoToCCF(probe_locs_SC[:,1])
        
        corrected_probe_locs_list_SC[i, :, :] = probe_locs_SC
        corrected_probe_locs_list_CCF[i, :, :] = probe_locs_CCF

    return corrected_probe_locs_list_SC, corrected_probe_locs_list_CCF


def reconstruct_shank_id(clusters_df, mouse, colname='unit_location_x'):
    # M21 was implanted in reverse: remap probe_x values so shank 0 is at the
    # left (low x) and shank 3 at the right (high x), matching all other mice.
    # The 8 unique probe_x values and their flipped counterparts:
    _M21_X_REMAP = {
        0.: 782., 32.: 750.,
        250.: 532., 282.: 500.,
        500.: 282., 532.: 250.,
        750.: 32.,  782.: 0.,
    }

    if mouse == 21:
        unique_before = sorted(clusters_df[colname].dropna().unique())
        print(f"M21 unique {colname} values (before remap): {unique_before}")
        clusters_df = clusters_df.copy()
        clusters_df[colname] = clusters_df[colname].apply(
            lambda v: _M21_X_REMAP.get(float(v), v)
        )
        unique_after = sorted(clusters_df[colname].dropna().unique())
        print(f"M21 unique {colname} values (after remap):  {unique_after}")

    shank_ids = []
    for index, cluster in clusters_df.iterrows():
        x_pos = cluster[colname]
        if x_pos <= 150:
            shank_id = 0
        elif x_pos <= 400:
            shank_id = 1
        elif x_pos <= 650:
            shank_id = 2
        else:
            shank_id = 3
        shank_ids.append(shank_id)
    clusters_df['shank_id'] = shank_ids
    return clusters_df



def brain_coord_from_xy(x_pos, y_pos, probe_locs_list_SC, shank_id):
    direction_vector = probe_locs_list_SC[shank_id,:,0] - probe_locs_list_SC[shank_id,:,1]
    # Calculate the unit vector
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    # Calculate the position
    brain_coord_SC = probe_locs_list_SC[shank_id,:,1] + (y_pos * unit_vector)
    brain_coord_SC[2] -= x_pos
    brain_coord_CCF = StereoToCCF(brain_coord_SC)
    z_CCF,y_CCF,x_CCF = np.round(brain_coord_CCF/10).astype(int)

    return brain_coord_SC, brain_coord_CCF



def get_annotation_colors(cluster_annotations):
    annotation_colors = []
    for i in range(len(cluster_annotations)):
        if 'ENT' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (106, 202,71)
        elif 'VIS' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (255, 255,71)
        elif 'RSP' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (0, 0 , 255)
        elif 'SUB' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (0, 255,255)
        elif 'PAR' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (45, 160,23)
        elif 'PRE' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (20, 130,83)
        elif 'HPF' in cluster_annotations[i]:
            color = '#%02x%02x%02x' % (0, 255, 0)
        else:
            #print(f'I couldnt assign a color for {cluster_annotations[i]}, so I made it red')
            color = '#%02x%02x%02x' % (255, 0, 0)
        annotation_colors.append(color)
    return annotation_colors


def get_annotation_colors(cluster_annotations):
    annotation_colors = []
    for i in range(len(cluster_annotations)):
        if 'ENT' in cluster_annotations[i]:
            color = (101, 196,165, 255)
        elif 'VIS' in cluster_annotations[i]:
            color = (231, 138,195, 255)
        elif 'RSP' in cluster_annotations[i]:
            color = (0, 0 , 255, 255)
        elif 'SUB' in cluster_annotations[i]:
            color = (0, 255,255, 255)
        elif 'PAR' in cluster_annotations[i]:
            color = (82, 187,210, 255)
        elif 'PRE' in cluster_annotations[i]:
            color = (20, 130, 83, 255)
        elif 'HPF' in cluster_annotations[i]:
            color = (0, 255, 0, 255)
        else:
            #print(f'I couldnt assign a color for {cluster_annotations[i]}, so I made it red')
            color = (255, 255, 255, 255)
        annotation_colors.append(color)


def make_annotations(mouse, path=None):
    """Build (or load from cache) a 2-D annotation array and matching color array
    for a given mouse, covering the probe x/y display space.

    Returns
    -------
    annotations       : np.ndarray, shape (len(ys), len(xs)), dtype=object  — brain-region acronym per pixel
    annotation_colors : np.ndarray, shape (len(ys), len(xs))               — hex color per pixel
    xs                : np.ndarray  — x-axis values (µm, probe space)
    ys                : np.ndarray  — y-axis values (µm, probe space)
    """
    import os
    from spatial_manifolds.detect_grids import get_annotation_colors_2D

    ymin = -300; ymax = 3000
    xmin = -200; xmax = 1000
    xs = np.arange(xmin, xmax, 1)
    ys = np.arange(ymin, ymax, 1)

    ann_path   = f"{path}/M{mouse}_annotations.npy"
    color_path = f"{path}/M{mouse}_annotation_colors.npy"

    if os.path.exists(ann_path) and os.path.exists(color_path):
        annotations       = np.load(ann_path,   allow_pickle=True)
        annotation_colors = np.load(color_path, allow_pickle=True)
        return annotations, annotation_colors, xs, ys

    # ── Build from probe registration files ───────────────────────────────
    data_paths = [
        f"/Users/harryclark/Documents/brainrender/probe_data/M{mouse}_probe_locations_{a}.mat"
        for a in [1, 2, 3, 4]
    ]
    shank_offsets_SC = pd.read_csv(
        '/Users/harryclark/Documents/brainrender/probe_data/shank_offsets.csv'
    )
    shank_offsets_SC = shank_offsets_SC[shank_offsets_SC['mouse'] == mouse]

    probes_locs = [read_probe_mat(p) for p in data_paths]
    adjusted_probes_locs_CCF = np.array([adjust_probe_locs(pl) for pl in probes_locs])
    adjusted_probe_locs_SC, adjusted_probe_locs_CCF = correct_for_left_side(adjusted_probes_locs_CCF)
    adjusted_probe_locs_SC, adjusted_probe_locs_CCF = adjust_to_shank_offsets(
        adjusted_probe_locs_SC, shank_offsets_SC
    )

    annotations = np.zeros((len(ys), len(xs)), dtype=object)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            coord_SC, coord_CCF = brain_coord_from_xy(x, y, adjusted_probe_locs_SC, shank_id=0)
            z_CCF, y_CCF, x_CCF = np.round(coord_CCF / 10).astype(int)
            annotation_index = annotations_set[z_CCF, y_CCF, x_CCF]
            matches = structure_set[structure_set['id'] == annotation_index]
            annotations[yi, xi] = matches['acronym'].iloc[0] if len(matches) == 1 else 'root'

    annotation_colors = get_annotation_colors_2D(annotations)

    np.save(ann_path,   annotations)
    np.save(color_path, annotation_colors)
    print(f"  [make_annotations] Saved to {path}")

    return annotations, annotation_colors, xs, ys
    return annotation_colors