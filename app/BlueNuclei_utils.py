import numpy as np 
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.interpolate import splprep, splev
import czifile
from scipy.signal import find_peaks, peak_widths, peak_prominences, find_peaks_cwt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
import pandas as pd
from skimage import io
from skimage import filters
from skimage import img_as_ubyte #for saving, and manipulating size, gamma correction
from skimage import img_as_float #for filtering
from skimage.filters import gaussian
from skimage.filters import median
from skimage import data, feature
from skimage import segmentation
from skimage import draw
from skimage.filters import threshold_triangle, threshold_otsu
from math import pi
from sklearn.cluster import DBSCAN
import cv2, __main__, scipy, re, os, math, skimage, sys, time, logging
import tempfile
import random
from matplotlib.path import Path


random.seed(42)

def filter_clusters_near_centroid(points, centroid, min_pts=3, weak_thresh=80, dist_threshold=15):
    clustering = DBSCAN(eps=2, min_samples=2).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    clusters = []

    for label in unique_labels:
        if label == -1:
            continue  # skip noise
        cluster_pts = points[labels == label]
        if len(cluster_pts) <= min_pts:
            continue  # discard very small clusters
        cluster_centroid = np.mean(cluster_pts, axis=0)
        dist = np.linalg.norm(cluster_centroid - centroid)

        other_dists = [
            np.linalg.norm(np.mean(points[labels == other], axis=0) - centroid)
            for other in unique_labels
            if other != -1 and other != label and len(points[labels == other]) > min_pts
        ]

        if len(cluster_pts) < weak_thresh and dist < dist_threshold:
            continue  # discard weak & close clusters

        clusters.append(cluster_pts)

    return np.vstack(clusters) if clusters else np.empty((0, 2))


def contour_coordinates(input_binary, min_area, cir_thre):
    #this function detects contours and coordinates of alreadly binarized (thresholded) image
    #input_binary is e.g. GFP_binary_triangle, or GFP_binary_otsu
    #min_area, cir_thre filters contours based on area and circularity
    #rx, ry scales image display window. try rx=ry=0.2 for large czi images.
    #show: either True (you want to display the contours), or False (just let this function return output)
    contours = cv2.findContours(input_binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    #contours[0] is collection of all coordinates, [1] is collection of hierachies (e.g. contour within a contour)
    #CHAIN_APPROX_NONE stores all points while CHAIN_APPROX_SIMPLE stores minimal points that defines the contour
    contours = contours[0] if len(contours) == 2 else contours[1]
    #now got rid of hierachy, 
    #contours contain only coordinates. e.g. contours[10000] means the 10000th closed contour's (x,y) coordinates
    #so far, contours collects all the contour coordinates, without filtering, e.g. ~20000 contours in one GFP image
    count=0
    contours_qualified=[]
    #apply size and circularity filters
    for c in contours:
        area = cv2.contourArea(c)
        if area == 0:
            continue
        perimeter = cv2.arcLength(c,True) 
        circularity = 4*math.pi*area/perimeter**2
        if min_area < area and circularity > cir_thre:
            count+=1
            contours_qualified.append(c)
    
    return(contours_qualified)


def contour_extreme(c):
    #c is the input coordinates set of a contour
    #this function extracts and returns the extreme points bounding an input contour
    #output of this function: ouput[0]=extLeft, output[1]=extRight, output[2]=extTop, output[3]=extBottom
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBottom = tuple(c[c[:, :, 1].argmax()][0])
    return([extLeft, extRight, extTop, extBottom])


def contour_refine(contours_1, contours_2):
    #pipe the output of double thresholding + contouring into this function
    #i.e.contours_1 and contour_2 are outputs from contour_coordinates function
    #contours_1 must be from low threshold (more contours), contours_2 must be from high threshold (less contours)

    shrink=[]
    repick=[]
    faint=contours_1
    drop_list=[]

    for ct2 in contours_2:
        ext = contour_extreme(ct2)
        x=0
        k=-1
        for ct1 in contours_1:
            k+=1
            #shrinking bright neurons: these are neurons detected in both contours_1 and contour_2
            if not (cv2.pointPolygonTest(ct1, (int(ext[0][0]),int(ext[0][1])), False) >= 0
                and cv2.pointPolygonTest(ct1, (int(ext[1][0]),int(ext[1][1])), False) >= 0
                and cv2.pointPolygonTest(ct1, (int(ext[2][0]),int(ext[2][1])), False) >= 0
                and cv2.pointPolygonTest(ct1, (int(ext[3][0]),int(ext[3][1])), False) >= 0):
                continue
            else:
                shrink.append(ct2)
                drop_list.append(k)
                x=1        
                break
        if x==0: #which means that this ct2 from contours_2 is not within any contours in contours_1
            #re-pick neurons filtered out in contours_1 due to circularity: these are neurons only detected in contours_2
            repick.append(ct2)
    faint_1 = [arr for i, arr in enumerate(faint) if i not in drop_list]
    # Apply swelling factors
    swell_factors = {
        'shrink': 5,
        'repick': 5,
        'faint_1': 1.5
    }


    swelled_shrink = [np.array(Polygon(np.squeeze(ct)).buffer(swell_factors['shrink']).exterior.coords, dtype=int) for ct in shrink]
    swelled_repick = [np.array(Polygon(np.squeeze(ct)).buffer(swell_factors['repick']).exterior.coords, dtype=int) for ct in repick]
    swelled_faint_1 = [np.array(Polygon(np.squeeze(ct)).buffer(swell_factors['faint_1']).exterior.coords, dtype=int) for ct in faint_1]

    # Combine swelled contours into a single list
    all_neu_con_swelled = swelled_shrink + swelled_repick + swelled_faint_1

    return all_neu_con_swelled


def draw_contour(contours, canvas, mask_exist, colors,
                 show_img=False, save_img=None, fp="",
                 text_labels=None, text_colors=None, label_positions=None):
    if not mask_exist:
        canvas = np.zeros_like(canvas, dtype=np.uint8)
    arrow_length = max(10, canvas.shape[0] // 150)     
    arrow_thickness = max(2, canvas.shape[0] // 1500)   
    arrow_tip_len = 0.3                                # tip size stays proportional


    for i, contour in enumerate(contours):
        if contour.ndim == 2:
            contour = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(canvas, [contour], -1, colors[i], 1)

    if text_labels and label_positions and text_colors:
        for text, pos, clr in zip(text_labels, label_positions, text_colors):
            arrow_tip = (int(pos[0]), int(pos[1])-30)
            arrow_base = (arrow_tip[0], arrow_tip[1] - arrow_length)
            cv2.arrowedLine(canvas,
                            arrow_base, arrow_tip,
                            color=(255, 0, 0),
                            thickness=arrow_thickness,
                            tipLength=arrow_tip_len)

            for line_idx, line in enumerate(text.split('\n')):
                text_pos = (int(pos[0]) + 20, int(pos[1]) + 20 + 22 * line_idx)
                cv2.putText(canvas, line, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 1)

    if show_img:
        Image.fromarray(canvas).show()
    if save_img is not None:
        Image.fromarray(canvas).save(os.path.join(fp, save_img))

    return canvas
       

def slice_int(input_table, sort_key, val_col): ###pipe this output directly into slice_max
    df = input_table.sort_values(by=sort_key, ascending = True)
    cor = df[sort_key[0]].tolist()
    int_val = df[val_col].tolist()
    global_diff = max(int_val)-min(int_val)
    cor_unique = list(dict.fromkeys(cor)) ###collects all unique x coordinates
    cor_range = []
    for c in cor_unique:
        cor_range.append(cor.index(c)) ###collects order indices of all unique x coordinates
    if sort_key[0] == "X":
        x_or_y_list = df[sort_key[1]].tolist()
    if sort_key[0] == "Y":
        x_or_y_list = df[sort_key[0]].tolist()
    centroid = [input_table['X'].mean(), input_table['Y'].mean()]
    # print("input_table:",input_table)
    # print("df:",df)
    # print("cor:",cor)
    # print("int_val:",int_val)
    # print("global_dif:",global_diff)
    # print("cor_unique:",cor_unique)
    # print("cor_range:",cor_range)
    # print('x_or_y_list:',x_or_y_list)
    # print('centroid:',centroid)
    w = [int_val, cor_range, cor_unique, global_diff, x_or_y_list, centroid]
    return(w) 

    
def slice_max(w,sort_key): ###pipe this output directly into spotty. w is what slice_int returns
    spt_final = 0
    distribution = 0
    cor_list = []
    distances = []
    for n, i in enumerate(w[1]): 
        spt = 0
        cor_1 = w[2][n]
        if n < len(w[1])-1:
            peak_range = w[0][w[1][n]:w[1][n+1]]
            x_or_y_range = w[4][w[1][n]:w[1][n+1]]
        else:
            peak_range = w[0][w[1][n]:]
            x_or_y_range = w[4][w[1][n]:]
        if len(peak_range) <= 5:
            continue
        
        peaks = find_peaks(np.array(peak_range), width=1,distance=1,prominence=0.01)
        for p in peaks[0]:
            cor_2 = x_or_y_range[p]
            if sort_key == "X":
                cor_list.append([cor_1, cor_2])
            elif sort_key == "Y":
                if [cor_2, cor_1] not in cor_list:
                    cor_list.append([cor_2, cor_1])
        peak_hei = np.asarray(peak_range)[peaks[0]].tolist() ###just the heights, no index
    #     print("sign changes:",len(peaks[0]))
    #     print("all_max:",all_max)
    #     print("peak hei:",peak_hei)
    #     print("peak indice:",np.asarray(w[2])[peaks[0]])
    #     print("corresponding peaks in all_max:",np.asarray(all_max)[peaks[0]])
        if len(peak_hei) <= 1: ###ie. if there is only one peak 
            spt_final += random.uniform(-1, -5)
            continue
        else:
            q = -1
            for each_peak in peak_hei:
                q += 1
                if q == 0:
                    spt += (each_peak-min(peak_range[peaks[0][0]:peaks[0][q+1]]))*2
                elif q < len(peak_hei)-1:
                    spt += (((each_peak-min(peak_range[peaks[0][q-1]:peaks[0][q]]))+(each_peak-min(peak_range[peaks[0][q]:peaks[0][q+1]])))/2)*2
                elif q == len(peak_hei)-1:
                    spt += (each_peak-min(peak_range[peaks[0][q-1]:peaks[0][-1]]))*2
            spt_final += len(peaks[0])*spt/w[3]
    if spt_final <= 0:
        spt_final = random.uniform(-1, -5)
    else:
        spt_final=math.log(spt_final)
    # Amplify distant peaks by non-linear weighting
    for spot in cor_list:
        d = math.sqrt((spot[0] - w[5][0])**2 + (spot[1] - w[5][1])**2)
        distances.append(d)

    if len(distances) > 0:
        distances = np.array(distances)
        distribution = np.sum(distances) 
        distribution = math.log(distribution)
    else:
        distribution = random.uniform(-1, -5)

    return spt_final, distribution, cor_list

def spot_distri_final(input_table): #returns a spottiness value and a distribution value
    return(slice_max(slice_int(input_table,['X','Y'],'Value'),"X")+slice_max(slice_int(input_table,['Y','X'],'Value'),"Y")) 

   
def calculate_norm(image):
    norm = np.bincount(image.ravel()).argmax()
    return norm


def roi_metrics(single_roi, full_laplacian, edge_scale=0.6):
    # === Spottiness & Distribution ===
    spottiness = spot_distri_final(single_roi)[0]
    distribution = spot_distri_final(single_roi)[1]

    # === Central Intensity ===
    points = list(zip(single_roi['X'], single_roi['Y']))
    poly = Polygon(points).convex_hull
    centroid = poly.centroid
    shrunken_poly = scale(poly, xfact=0.4, yfact=0.4, origin=centroid)
    inside_mask = [shrunken_poly.contains(Point(x, y)) for x, y in points]
    shrunken_values = single_roi['Value'][inside_mask]
    intensity = shrunken_values.mean() if len(shrunken_values) > 0 else 0

    # === Area ===
    area = len(single_roi)

    # === Edge Gradient using Shapely-scaled belt ===
    try:
        # Use convex hull of ROI
        original_poly = poly
        inner_poly = scale(original_poly, xfact=edge_scale, yfact=edge_scale, origin=centroid)
        belt_region = original_poly.difference(inner_poly)

        # Identify points in the belt region
        belt_mask = [belt_region.contains(Point(x, y)) for x, y in zip(single_roi['X'], single_roi['Y'])]

        # Sample Laplacian at those belt points
        if np.any(belt_mask):
            ys = single_roi['Y'][belt_mask].values
            xs = single_roi['X'][belt_mask].values
            # Clip coordinates to stay in bounds
            ys = np.clip(ys, 0, full_laplacian.shape[0] - 1)
            xs = np.clip(xs, 0, full_laplacian.shape[1] - 1)
            edge_gradient = full_laplacian[ys, xs].mean()
        else:
            edge_gradient = 0

    except Exception as e:
        edge_gradient = 0
        print({e})

    return pd.DataFrame([{
        'spottiness': spottiness,
        'distribution': distribution,
        'intensity': intensity,
        'area': area,
        'edge_gradient': edge_gradient
    }])
 

def smoothed_pts_to_dataframe(smoothed_pts, img4_DAPI):
    if smoothed_pts.size == 0:
        return pd.DataFrame(columns=['X', 'Y', 'Value'])
    rounded = np.round(smoothed_pts).astype(int)
    h, w = img4_DAPI.shape
    in_bounds = (rounded[:, 0] >= 0) & (rounded[:, 0] < w) & \
                (rounded[:, 1] >= 0) & (rounded[:, 1] < h)
    rounded = rounded[in_bounds]
    unique_pts = np.unique(rounded, axis=0)
    values = [img4_DAPI[y, x] for x, y in unique_pts]
    df = pd.DataFrame(unique_pts, columns=['X', 'Y'])
    df['Value'] = values
    return df

def select_best_polygon(points, candidate_centers, circularity_thresh=0.5):
    best_poly = np.empty((0, 2), dtype=int)
    best_score = -np.inf
    for c in candidate_centers:
        poly = get_inner_ring(points, center=c)
        circ = polygon_circularity(poly)
        area = polygon_area(poly)
        if circ > circularity_thresh and area > 300:
            score = circ * area
            if score > best_score:
                best_score = score
                best_poly = poly
    return best_poly

def is_point_connected(pt, point_set):
    return any((pt[0]+dx, pt[1]+dy) in point_set for dx in [-1, 0, 1] for dy in [-1, 0, 1])

def get_inner_ring(points, center=None, num_sectors=38):
    if len(points) == 0:
        return np.empty((0, 2), dtype=int)
    if center is None:
        center = np.mean(points, axis=0)
    shifted = points - center
    angles = np.arctan2(shifted[:, 1], shifted[:, 0])
    distances = np.linalg.norm(shifted, axis=1)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    inner_points = []
    for i in range(num_sectors):
        a0 = (i / num_sectors) * 2 * np.pi
        a1 = ((i + 1) / num_sectors) * 2 * np.pi
        mask = (angles >= a0) & (angles < a1)
        if np.any(mask):
            sector_pts = points[mask]
            sector_dists = distances[mask]
            idx = np.argmin(sector_dists)
            inner_points.append(sector_pts[idx])
    return np.array(inner_points)

def find_better_centroid(base_centroid, point_set, max_radius=30, step=10):
    directions = np.array([[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])
    for radius in [step, max_radius]:
        candidates = base_centroid + directions * radius
        candidates = np.round(candidates).astype(int)
        good_centers = [cand for cand in candidates if tuple(cand) not in point_set]
        if good_centers:
            return good_centers
    return []

def polygon_area(pts):
    if len(pts) < 3: return 0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_circularity(pts):
    area = polygon_area(pts)
    if area == 0: return 0
    perimeter = np.sum(np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1))
    return 4 * pi * area / (perimeter ** 2)

def area_poly(pts):
    if len(pts) < 3: return 0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def autoscale_image(image, lower_percentile=0.25, upper_percentile=99.75):
    p_low, p_high = np.percentile(image, (lower_percentile, upper_percentile))
    if p_high == p_low:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((image - p_low) / (p_high - p_low), 0, 1)
    return img_as_ubyte(scaled)


def sanitize_title_for_filename(title: str) -> str:
    # Replace non-alphanumeric characters with underscores
    title = re.sub(r'[^A-Za-z0-9]', '_', title)
    # Collapse multiple underscores
    title = re.sub(r'_{2,}', '_', title)
    # Strip leading/trailing underscores
    return title.strip('_')


def process_single_image(fp, model, std_scaler, minmax_scaler, thr=-2.8,
                            visual=True, plotdir=None, master_crop=None, 
                            debug=False, debug_save_plots=False, debug_contour_idx=None):
    try:
        start_time = time.time()
        outdir = os.path.join(tempfile.gettempdir(), os.path.splitext(os.path.basename(fp))[0])
        
        os.makedirs(outdir, exist_ok=True)
        print("Saving debug plots to:", plotdir)
        print("Saving visual plots to:", outdir)


        def load_czi_channels(fp):
            with czifile.CziFile(fp) as czi:
                arr = czi.asarray()
                # Typical shape: (T, C, Z, Y, X, 1)
                arr = arr.squeeze()  # Remove singleton dims
        
                # Assume arr now has shape (C, Y, X)
                if arr.ndim == 4:
                    # Fallback: (C, Z, Y, X) → use Z=0
                    img4_GFP = arr[1, 0, :, :]
                    img4_DAPI = arr[0, 0, :, :]
                elif arr.ndim == 3:
                    # (C, Y, X)
                    img4_GFP = arr[1, :, :]
                    img4_DAPI = arr[0, :, :]
                else:
                    raise ValueError(f"Unexpected image shape: {arr.shape}")
                return img4_GFP, img4_DAPI
        img4_GFP, img4_DAPI = load_czi_channels(fp)

        # Optional cropping (master_crop should be a tuple like ((x0, y0), (x1, y1)))

        if master_crop:
            (x0, y0), (x1, y1) = master_crop
            img4_GFP = img4_GFP[y0:y1, x0:x1]
            img4_DAPI = img4_DAPI[y0:y1, x0:x1]

        if debug:
            # Plot merged: DAPI (Blue), GFP (Green)
            plt.figure(figsize=(7, 7))
            merged = np.zeros((*img4_DAPI.shape, 3), dtype=np.uint8)
            merged[..., 1] = autoscale_image(img4_GFP)  # Green channel
            merged[..., 2] = autoscale_image(img4_DAPI)  # Blue channel
            plt.imshow(merged)
            title_obj = plt.title("Original Image: Merged")
            plt.axis("off")
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()


            # Plot individual DAPI (Blue on black) and GFP (Green on black)
            plt.figure(figsize=(7, 7))
            # plt.subplot(1, 2, 1)
            dapi_blue = np.zeros((*img4_DAPI.shape, 3), dtype=np.uint8)
            dapi_blue[..., 2] = merged[..., 2]
            plt.imshow(dapi_blue)
            title_obj = plt.title("Original Image: DAPI")
            plt.axis("off")
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()


            plt.figure(figsize=(7, 7))
            # plt.subplot(1, 2, 2)
            gfp_green = np.zeros((*img4_GFP.shape, 3), dtype=np.uint8)
            gfp_green[..., 1] = merged[..., 1]
            plt.imshow(gfp_green)
            title_obj = plt.title("Original Image: GFP")
            plt.axis("off")
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()


    except Exception as e:
        logging.error(f"Failed to load image: {fp}", exc_info=True)
        return {"filename": os.path.basename(fp), "status": f"❌ Failed to load image: {e}"}
    try:
        # Thresholding
        thresh_triangle = threshold_triangle(img4_GFP)
        binary_triangle = img4_GFP > thresh_triangle
        GFP_binary_triangle = binary_triangle.astype(np.uint8)

        thresh_otsu = threshold_otsu(img4_GFP) * 0.7
        binary_otsu = img4_GFP > thresh_otsu
        GFP_binary_otsu = binary_otsu.astype(np.uint8)

        if debug:
            plt.figure(figsize=(7, 7))
            # plt.subplot(1, 2, 1)
            plt.imshow(GFP_binary_triangle, cmap='gray')
            title_obj = plt.title("GFP Channel: Triangular Thresholding")
            plt.axis("off")
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()
            

            plt.figure(figsize=(7, 7))
            # plt.subplot(1, 2, 2)
            plt.imshow(GFP_binary_otsu, cmap='gray')
            title_obj = plt.title("GFP Channel: Otsu Thresholding")
            plt.axis("off")
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()


        # Contour extraction
        all_neu_con = contour_refine(
            contour_coordinates(GFP_binary_triangle, 400, 0.10),
            contour_coordinates(GFP_binary_otsu, 200, 0.10)
        )

        if debug:
            for binary, title, area_thres in zip(
                [GFP_binary_triangle, GFP_binary_otsu],
                ["Triangular", "Otsu"],
                [400, 200]  # same thresholds as used in contour_coordinates call
            ):
                filtered_contours = contour_coordinates(binary, area_thres, 0.1)
                rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
                rgb[..., 1] = merged[..., 1]

                plt.figure(figsize=(7, 7))
                plt.imshow(rgb)
                title_obj = plt.title(f"Neurons Detected via {title} Thresholding")
                plt.axis("off")

                first = True
                for con in filtered_contours:
                    if con.ndim == 3:
                        con = con.reshape(-1, 2)
                    if first:
                        plt.plot(con[:, 0], con[:, 1], color='red', linewidth=0.8, label='Detected Neurons')
                        first = False
                    else:
                        plt.plot(con[:, 0], con[:, 1], color='red', linewidth=0.8)

                plt.legend(fontsize=8,loc='upper right')

                if debug_save_plots:
                    # Get the actual title string
                    plot_title = title_obj.get_text()
                    safe_title = sanitize_title_for_filename(plot_title)
                    # Save the plot
                    filename = f"{safe_title}.png"
                    filepath = os.path.join(plotdir, filename)
                    plt.savefig(filepath, dpi=1200, bbox_inches="tight")
                plt.show()

        if debug:
            rgb = np.zeros((*img4_DAPI.shape, 3), dtype=np.uint8)
            rgb[..., 1] = merged[..., 1]  # background (e.g., green channel)
            plt.figure(figsize=(7, 7))
            plt.imshow(rgb)
            title_obj = plt.title("All Neurons Detected and Size-Refined")
            plt.axis("off")

            first = True
            for con in all_neu_con:
                if con.ndim == 3:
                    con = con.reshape(-1, 2)
                poly = Polygon(con)
                # Enlarge contours for better visualization
                enlarged_poly = scale(poly, xfact=2, yfact=2, origin='centroid')
                enlarged_coords = np.array(enlarged_poly.exterior.coords)
                # Plot with legend on first
                if first:
                    plt.plot(enlarged_coords[:, 0], enlarged_coords[:, 1], color='red', linewidth=0.8, label='Detected Neurons')
                    first = False
                else:
                    plt.plot(enlarged_coords[:, 0], enlarged_coords[:, 1], color='red', linewidth=0.8)

            plt.legend(fontsize=8,loc='upper right')
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()



        blurred = cv2.GaussianBlur(img4_DAPI, (27, 27), 0)
        if debug:
            plt.figure(figsize=(7, 7))
            blur_rgb = np.zeros((*blurred.shape, 3), dtype=np.uint8)
            blur_rgb[..., 2] = autoscale_image(blurred)
            plt.imshow(blur_rgb)
            title_obj = plt.title("DAPI Channel: Gaussian Blur")
            plt.axis("off")
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.show()


        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

        binary = (laplacian > 0).astype(np.uint8)

        bck_DAPI = np.bincount(img4_DAPI.ravel()).argmax()
        median_DAPI = np.median(img4_DAPI)

        small_collections = []
        all_nuc_con = []
        enlarged_all_neu_con = []
        roi_to_neuron_idx = []
    
        for idx_neuron, original_points in enumerate(all_neu_con):
            area = area_poly(original_points)
            if area < 600:
                scl = 3.5
            elif area < 1100:
                scl = 3.5 - (area - 600) * (2.5 / 500.0)
            elif area < 1700:
                scl = 2.5 - (area - 1100) * (1.0 / 600.0)
            elif area > 2500:
                scl = 0.8
            else:
                scl = 1.0 - (area - 1700) * (0.2 / 800.0)
            scl = np.sqrt(scl)

            cx, cy = np.mean(original_points[:, 0]), np.mean(original_points[:, 1])
            enlarged = (original_points - [cx, cy]) * scl + [cx, cy]
            enlarged_all_neu_con.append(enlarged.astype(np.int32))
            points = enlarged.astype(int)

            x0, x1 = max(points[:,0].min(), 0), min(points[:,0].max(), binary.shape[1])
            y0, y1 = max(points[:,1].min(), 0), min(points[:,1].max(), binary.shape[0])

            binary_crop = binary[y0:y1, x0:x1]
            offset = points - [x0, y0]
            rr, cc = polygon(offset[:, 1], offset[:, 0], binary_crop.shape)
            mask = np.zeros_like(binary_crop, dtype=bool)
            mask[rr, cc] = True

            edge_mask = binary_crop.astype(bool) & mask
            coords = np.argwhere(edge_mask)
            selected_xy = coords[:, ::-1] + [x0, y0]

            lap_values = laplacian[selected_xy[:, 1], selected_xy[:, 0]]
            strong = selected_xy[lap_values > 5]
            strong = strong[img4_DAPI[strong[:,1], strong[:,0]] > 2 * bck_DAPI]
            

            if len(strong) == 0:
                continue

            vals = img4_DAPI[strong[:, 1], strong[:, 0]]
            hist, bins = np.histogram(vals, bins=50)
            if hist[0] + hist[1] < 0.3 * hist.sum():
                strong = strong[vals > bins[2]]

            centroid = np.mean(strong, axis=0)
            rounded = tuple(np.round(centroid).astype(int))
            point_set = set(map(tuple, strong))

            if not is_point_connected(rounded, point_set):
                clean = filter_clusters_near_centroid(strong, centroid)
                poly = get_inner_ring(clean, center=centroid)
            else:
                centers = find_better_centroid(centroid, point_set)
                if centers:
                    poly = select_best_polygon(strong, centers)
                else:
                    clean = filter_clusters_near_centroid(strong, centroid)
                    poly = get_inner_ring(clean, center=centroid)

            smoothed_pts = np.empty((0, 2))
            if poly.shape[0] >= 4:
                try:
                    tck, _ = splprep([poly[:, 0], poly[:, 1]], s=10.0, per=True)
                    fine = np.linspace(0, 1, 500)
                    x_smooth, y_smooth = splev(fine, tck)
                    temp = np.vstack((x_smooth, y_smooth)).T
                    if polygon_area(temp) > 300 and polygon_circularity(temp) > 0.50:
                        smoothed_pts = temp
                except:
                    temp = poly.astype(float)
                    if polygon_area(temp) > 300 and polygon_circularity(temp) > 0.50:
                        smoothed_pts = temp

            if smoothed_pts.size > 0:
                nuc_contour = np.round(smoothed_pts).astype(np.int32).reshape(-1, 1, 2)
                all_nuc_con.append(nuc_contour)
                poly = np.round(smoothed_pts).astype(np.int32)
                rr, cc = polygon(poly[:, 1], poly[:, 0], img4_DAPI.shape)
                rr, cc = rr.clip(0, img4_DAPI.shape[0]-1), cc.clip(0, img4_DAPI.shape[1]-1)
                vals = img4_DAPI[rr, cc]
                df_roi = pd.DataFrame({'X': cc, 'Y': rr, 'Value': vals})
            else:
                df_roi = pd.DataFrame(columns=['X', 'Y', 'Value'])

            if not df_roi.empty:
                small_collections.append(df_roi)
                roi_to_neuron_idx.append(idx_neuron)

            if debug and debug_contour_idx==idx_neuron and smoothed_pts.size > 0:
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                y_min, y_max = points[:, 1].min(), points[:, 1].max()

                padding = 15
                x0_viz = max(x_min - padding, 0)
                x1_viz = min(x_max + padding, binary.shape[1])
                y0_viz = max(y_min - padding, 0)
                y1_viz = min(y_max + padding, binary.shape[0])

                viz_crop = img4_DAPI[y0_viz:y1_viz, x0_viz:x1_viz]
                intensity_crop = img4_DAPI[y0_viz:y1_viz, x0_viz:x1_viz]

                dapi_rgb_crop = np.zeros((*viz_crop.shape, 3), dtype=np.uint8)
                dapi_rgb_crop[..., 2] = autoscale_image(viz_crop)  # DAPI → blue channel

                offset_points = points - [x0_viz, y0_viz]
                offset_filtered = strong - [x0_viz, y0_viz]
                offset_polygon = smoothed_pts - [x0_viz, y0_viz]

                polygon_path = Path(offset_polygon)
                yy, xx = np.meshgrid(np.arange(viz_crop.shape[0]), np.arange(viz_crop.shape[1]), indexing='ij')
                coords = np.vstack((xx.ravel(), yy.ravel())).T
                mask_inside = polygon_path.contains_points(coords).reshape(viz_crop.shape)

                inside_yy, inside_xx = np.where(mask_inside)
                intensity_inside_vals = intensity_crop[inside_yy, inside_xx]

                inner_ring = get_inner_ring(strong, center=centroid)
                offset_inner = inner_ring - [x0_viz, y0_viz]


                plt.figure(figsize=(7, 7))
                plt.imshow(dapi_rgb_crop)
                plt.scatter(offset_points[:, 0], offset_points[:, 1], color='red', s=5, label='Neuron')
                plt.scatter(offset_filtered[:, 0], offset_filtered[:, 1], c=[(1.0, 0.843, 0.0)], s=5, label='Candidate Points')
                title_obj = plt.title("Candidate Nuclear Edge Points (zoom-in)")
                plt.legend(fontsize=8,loc='upper right')
                plt.axis('off')
                if debug_save_plots:
                    # Get the actual title string
                    plot_title = title_obj.get_text()
                    safe_title = sanitize_title_for_filename(plot_title)
                    # Save the plot
                    filename = f"{safe_title}.png"
                    filepath = os.path.join(plotdir, filename)
                    plt.savefig(filepath, dpi=1200, bbox_inches="tight")
                plt.show()


                plt.figure(figsize=(7, 7))
                plt.imshow(dapi_rgb_crop)
                plt.scatter(offset_points[:, 0], offset_points[:, 1], color='red', s=5, label='Neuron')
                plt.scatter(offset_filtered[:, 0], offset_filtered[:, 1], s=5, c=[(1.0, 0.843, 0.0)], label='Candidate Points')
                plt.scatter([centroid[0] - x0_viz], [centroid[1] - y0_viz], c='magenta', s=100, marker='+', label='Centroid')
                plt.scatter(offset_inner[:, 0], offset_inner[:, 1], c=[(1.0, 0.5, 0.0)], s=50, marker='*', label='Closest Points to Centroid')
                title_obj = plt.title("Finding Nucleus")
                plt.legend(fontsize=8,loc='upper right')
                plt.axis('off')
                if debug_save_plots:
                    # Get the actual title string
                    plot_title = title_obj.get_text()
                    safe_title = sanitize_title_for_filename(plot_title)
                    # Save the plot
                    filename = f"{safe_title}.png"
                    filepath = os.path.join(plotdir, filename)
                    plt.savefig(filepath, dpi=1200, bbox_inches="tight")
                plt.show()

                plt.figure(figsize=(7, 7))
                plt.imshow(dapi_rgb_crop)
                plt.scatter(offset_points[:, 0], offset_points[:, 1], color='red', s=5, label='Neuron')
                plt.scatter(offset_filtered[:, 0], offset_filtered[:, 1], s=5, c=[(1.0, 0.843, 0.0)], label='Candidate Points')
                plt.scatter([centroid[0] - x0_viz], [centroid[1] - y0_viz], c='magenta', s=100, marker='+', label='Centroid')
                plt.scatter(offset_inner[:, 0], offset_inner[:, 1], c=[(1.0, 0.5, 0.0)], s=50, marker='*', label='Closest Points to Centroid')
                plt.plot(offset_polygon[:, 0], offset_polygon[:, 1], c=(0.5, 0.0, 0.7), linewidth=2, label='Fitted polygon')
                title_obj = plt.title("Spline-fitting")
                plt.legend(fontsize=8,loc='upper right')
                plt.axis('off')
                if debug_save_plots:
                    # Get the actual title string
                    plot_title = title_obj.get_text()
                    safe_title = sanitize_title_for_filename(plot_title)
                    # Save the plot
                    filename = f"{safe_title}.png"
                    filepath = os.path.join(plotdir, filename)
                    plt.savefig(filepath, dpi=1200, bbox_inches="tight")
                plt.show()

                plt.figure(figsize=(7, 7))
                plt.imshow(dapi_rgb_crop)
                plt.plot(offset_polygon[:, 0], offset_polygon[:, 1], color=(0.5, 0.0, 0.7), linewidth=2)  # Just the border
                title_obj = plt.title("Nucleus Found")
                plt.axis('off')

                if debug_save_plots:
                    # Get the actual title string
                    plot_title = title_obj.get_text()
                    safe_title = sanitize_title_for_filename(plot_title)
                    # Save the plot
                    filename = f"{safe_title}.png"
                    filepath = os.path.join(plotdir, filename)
                    plt.savefig(filepath, dpi=1200, bbox_inches="tight")
                plt.show()

                if debug and debug_contour_idx == idx_neuron:
                    lap_crop = laplacian[y0_viz:y1_viz, x0_viz:x1_viz]
                    lap_pos_crop = np.where(lap_crop > 0, lap_crop, 0)
                    
                    offset_points = points - [x0_viz, y0_viz]

                    plt.figure(figsize=(7, 7))
                    plt.imshow(np.clip(lap_pos_crop, 0, 50), cmap='cividis')
                    plt.scatter(offset_points[:, 0], offset_points[:, 1], color='red', s=5, label='Neuron Contour')
                    title_obj = plt.title("Laplacian Colormap (Cropped) with Neuron Contour")
                    plt.axis("off")
                    plt.legend(fontsize=8, loc='upper right')
                    cbar = plt.colorbar(shrink=0.7)
                    cbar.set_label("Laplacian Value", rotation=270, labelpad=15)

                    if debug_save_plots:
                        plot_title = title_obj.get_text()
                        safe_title = sanitize_title_for_filename(plot_title)
                        filename = f"{safe_title}.png"
                        filepath = os.path.join(plotdir, filename)
                        plt.savefig(filepath, dpi=1200, bbox_inches="tight")
                    plt.show()


        # === Laplacian colormap 
        if debug:
            laplacian_pos = np.where(laplacian > 0, laplacian, 0)
            plt.figure(figsize=(7, 7))
            plt.imshow(np.clip(laplacian_pos, 0, 50), cmap='cividis')
            title_obj = plt.title("Laplacian Colormap of Nuclei")
            plt.axis("off")
            cbar = plt.colorbar(shrink=0.7)
            cbar.set_label("Laplacian value", rotation=270, labelpad=15)
            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.legend(fontsize=8,loc='upper right')
            plt.show()


        # === Laplacian colormap and neuron contours
        if debug:
            laplacian_pos = np.where(laplacian > 0, laplacian, 0)

            plt.figure(figsize=(7, 7))
            plt.imshow(np.clip(laplacian_pos, 0, 50), cmap='cividis')
            title_obj = plt.title("Detected Neurons Overlaid\non Laplacian Colormap of Nuclei")
            plt.axis("off")
            cbar = plt.colorbar(shrink=0.7)
            cbar.set_label("Laplacian value", rotation=270, labelpad=15)

            first = True
            for con in all_neu_con:
                if con.ndim == 3:
                    con = con.reshape(-1, 2)
                poly = Polygon(con)
                # Enlarge contours for better visualization
                enlarged_poly = scale(poly, xfact=2, yfact=2, origin='centroid')
                enlarged_coords = np.array(enlarged_poly.exterior.coords)
                # Plot with legend on first
                if first:
                    plt.plot(enlarged_coords[:, 0], enlarged_coords[:, 1], color='red', linewidth=0.6, label='Neurons')
                    first = False
                else:
                    plt.plot(enlarged_coords[:, 0], enlarged_coords[:, 1], color='red', linewidth=0.6)

            if debug_save_plots:
                # Get the actual title string
                plot_title = title_obj.get_text()
                safe_title = sanitize_title_for_filename(plot_title)
                # Save the plot
                filename = f"{safe_title}.png"
                filepath = os.path.join(plotdir, filename)
                plt.savefig(filepath, dpi=1200, bbox_inches="tight")
            plt.legend(fontsize=8,loc='upper right')
            plt.show()


        # Extract features from all collected ROIs
        df_final = pd.DataFrame(columns=['spottiness', 'distribution', 'intensity', 'area', 'edge_gradient'])
        for roi in small_collections:
            row = roi_metrics(roi, laplacian)
            df_final = pd.concat([df_final, row])
        df_final['intensity'] = df_final['intensity'] / median_DAPI

        # === T → A domain adjustment for all 4 features ===
        features = ['spottiness', 'distribution', 'intensity', 'area', 'edge_gradient']

        # A-domain reference stats from training
        # spot, dist, inten, area, edge_grad
        A_means = np.array([-0.198422, 5.424109, 8.533572, 501.087912, 0.000375])  
        A_stds  = np.array([ 3.453235, 1.271679, 2.767525, 316.765321, 0.000289])

        # T-domain stats for this unseen image
        T_means = df_final[features].mean().values
        T_stds  = df_final[features].std().values

        # Prevent divide-by-zero
        T_stds_safe = np.where(T_stds == 0, 1, T_stds)

        # Apply feature-wise affine alignment
        aligned = ((df_final[features].values - T_means) / T_stds_safe) * A_stds + A_means
        df_final[features] = aligned


        scaled_df = df_final.copy()
        scaled_df[['spottiness', 'distribution', 'intensity', 'edge_gradient']] = std_scaler.transform(
            scaled_df[['spottiness', 'distribution', 'intensity', 'edge_gradient']])
        scaled_df[['area']] = minmax_scaler.transform(scaled_df[['area']])
        scaled = scaled_df.values

        # === Use decision function and apply best threshold ===
        confs = model.decision_function(scaled)  # raw decision values

        # Replace this with your actual saved threshold from training
        best_threshold = thr

        preds = (confs > best_threshold).astype(int)

        # Assign classes based on adjusted prediction
        scaled_df = pd.DataFrame(scaled, columns=features)
        scaled_df['class'] = ['live' if p == 1 else 'dead' for p in preds]

        try:
            confs = model.decision_function(scaled)
        except:
            confs = np.ones_like(preds) * 0.5

        count_live = (preds == 1).sum()
        count_dead = (preds == 0).sum()
        if visual == True: #For showing the interactive plots in GUI in the web app
            # Colors
            color_1 = (255, 255, 0)
            color_2 = (0, 255, 255)
            color_3 = (255, 0, 0)
            conf_thresh = 0.3

            # === Plot 1: neuron contours ===
            gfp_rgb = np.zeros((*img4_GFP.shape, 3), dtype=np.uint8)
            gfp_rgb[..., 1] = autoscale_image(img4_GFP)

            contours_1 = [np.array(c) for c in enlarged_all_neu_con]

            label_positions_1 = [np.mean(c.reshape(-1, 2), axis=0) for c in contours_1]
            text_labels_1 = ["neuron" for _ in contours_1]  # dummy text, ignored visually
            text_colors_1 = [(255, 0, 0)] * len(contours_1)

            draw_contour(contours_1, gfp_rgb, True, [color_1] * len(contours_1),
                         text_labels=text_labels_1,
                         text_colors=text_colors_1,
                         label_positions=label_positions_1,
                         show_img=False, save_img='plot1.png', fp=outdir)
            print("Saved:", os.path.join(outdir, 'plot1.png'))  # And same for plot2

            # === Plot 2: all nuclei + their neuron contours ===
            dapi_low = np.zeros((*img4_DAPI.shape, 3), dtype=np.uint8)
            dapi_low[..., 2] = autoscale_image(img4_DAPI)

            contours_2a = [np.array(c) for c in enlarged_all_neu_con]
            dapi_rgb = draw_contour(contours_2a, dapi_low, True, [color_1] * len(contours_2a))

            contours_2b = all_nuc_con
            labels = []
            label_positions = []
            text_colors = []
            draw_colors = []

            for i, roi in enumerate(small_collections):
                if roi.empty: continue
                raw_vals = df_final.iloc[i]
                center = (roi["X"].mean(), roi["Y"].mean())
                scaled_vals = scaled[i]  # These are the exact values passed to the SVM
                cls = "Live" if preds[i] == 1 else "Dead"
                conf = confs[i]
                col = color_2 if abs(conf) > conf_thresh else color_3
                label = (f"#{i} (scaled value/raw value)\n"
                         f"Area*: {scaled_vals[3]:.3f} / {raw_vals['area']:.0f}\n"
                         f"Intensity*: {scaled_vals[2]:.3f} / {raw_vals['intensity']:.2f}\n"
                         f"Spottiness*: {scaled_vals[0]:.3f} / {raw_vals['spottiness']:.2f}\n"
                         f"Distribution*: {scaled_vals[1]:.3f} / {raw_vals['distribution']:.2f}\n"
                         f"Edge_gradient*: {scaled_vals[4]:.3f} / {raw_vals['edge_gradient']:.2f}\n"
                         f"Class: {cls} ({conf:.2f})")


                labels.append(label)
                label_positions.append(center)
                text_colors.append(col)
                draw_colors.append(col)

            contours_2b = [c.reshape(-1, 2) for c in contours_2b]
            dapi_rgb = draw_contour(contours_2b, dapi_rgb, True, draw_colors,
                                    text_labels=labels, text_colors=text_colors,
                                    label_positions=label_positions, 
                                    show_img=False, save_img='plot2.png', fp=outdir)
            print("Saved:", os.path.join(outdir, 'plot2.png'))  # And same for plot2

           
        elapsed = time.time() - start_time
        return {
            'filename': os.path.basename(fp),
            'live': int(count_live),
            'dead': int(count_dead),
            'total': int(count_live + count_dead),
            'time_sec': round(elapsed, 2),
            'status': '✅ Success',
            'confidence': confs.tolist(),
            'predictions': preds.tolist(),
            'svm_input': scaled_df,
            'svm_input_raw': df_final.assign(**{'class': ['live' if p == 1 else 'dead' for p in preds]}),
            'plot_dir': os.path.basename(outdir),
            'total_neurons': len(all_neu_con),

        }

    except Exception as e:
        logging.error(f"Analysis failed for image: {fp}", exc_info=True)
        return {'filename': os.path.basename(fp), 'status': f'❌ Analysis failed: {e}'}