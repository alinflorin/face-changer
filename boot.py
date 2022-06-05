from flask import Flask, request, send_file
import flask
import json
import cv2
import os
import numpy as np
import dlib
import tempfile
import ffmpeg

# Indices for face landmarks
LOWER_HEAD = list(range(0, 17))
LEFT_BROW = list(range(17, 22))
RIGHT_BROW = list(range(22, 27))
NOSE = list(range(27, 36))
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

INDICES = {
    "lower_head": LOWER_HEAD,
    "left_brow": LEFT_BROW,
    "right_brow": RIGHT_BROW,
    "nose": NOSE,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "mouth": MOUTH
}


def detect_features(img, face_detector, feature_detector):
    """Detects facial features from a given image
    """
    features = []
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:

        for f in range(0, len(faces)):
            landmark_list = []
            face = faces[f]

            landmarks = feature_detector(image=gray, box=face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmark_list.append((x, y))

            # Mirror lower head around x-axis, in order to also include hairline, as landmarks only include chin otherwise
            highest_y = np.array(landmark_list)[
                LOWER_HEAD].min(axis=0)[1]
            for i in LOWER_HEAD:
                landmark_list.append(
                    (landmark_list[i][0], max(1, 2*highest_y-landmark_list[i][1])))
            features.append(np.array(landmark_list))
        return features
    else:
        return None


def zoom(points, factor):
    """Zoom the points by the given factor, with the anchor at the center.
    """
    center = np.mean(points, axis=0)
    return (points-center)*factor + center


def get_new_features(features, filter):
    """Apply filter to features
    """
    new_features = features.copy()

    for k, v in filter.items():
        indices = INDICES[k]
        new_features[indices] = zoom(
            new_features[indices], v["zoom"]) + v["trans"]

    return np.array(new_features)


def get_triangulation_indices(points):
    """Get indices triples for every triangle
    """
    # Bounding rectangle

    bounding_rect = (*points.min(axis=0), *points.max(axis=0))
    # Triangulate all points
    subdiv = cv2.Subdiv2D(bounding_rect)
    for p in points:
        try:
            subdiv.insert(tuple(map(float, p)))
        except Exception as ex:
            print(ex)
            pass
    # Iterate over all triangles
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        # Get index of all points
        yield [(points == point).all(axis=1).nonzero()[0][0] for point in [(x1, y1), (x2, y2), (x3, y3)]]


def crop_to_triangle(img, triangle):
    """Crop image to triangle
    """
    # Get bounding rectangle
    bounding_rect = cv2.boundingRect(triangle)
    # Crop image to bounding box
    img_cropped = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                      bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # Move triangle to coordinates in cropped image
    triangle_cropped = [
        (point[0]-bounding_rect[0], point[1]-bounding_rect[1]) for point in triangle]
    return triangle_cropped, img_cropped


def transform(src_img, src_points, dst_points):
    """Transforms source image to target image, overwriting the target image.
    """
    dst_img = src_img.copy()

    for indices in get_triangulation_indices(src_points):
        try:
            # Get triangles from indices
            src_triangle = src_points[indices]
            dst_triangle = dst_points[indices]

            # Crop to triangle, to make calculations more efficient
            src_triangle_cropped, src_img_cropped = crop_to_triangle(
                src_img, src_triangle)
            dst_triangle_cropped, dst_img_cropped = crop_to_triangle(
                dst_img, dst_triangle)

            # Calculate transfrom to wrap from old image to new
            transform = cv2.getAffineTransform(np.float32(
                src_triangle_cropped), np.float32(dst_triangle_cropped))

            # Warp image
            dst_img_warped = cv2.warpAffine(src_img_cropped, transform, (
                dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Create mask for the triangle we want to transform
            mask = np.zeros(dst_img_cropped.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(
                dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0)

            # Delete all existing pixels at given mask
            dst_img_cropped *= 1-mask
            # Add new pixels to masked area
            dst_img_cropped += dst_img_warped*mask

        except Exception as e:
            pass
    return dst_img


app = Flask(__name__)

# Set up face detector
face_detector = dlib.get_frontal_face_detector()
feature_detector = dlib.shape_predictor(
   "shape_predictor_68_face_landmarks.dat")


@app.route("/api/filters/apply", methods=['POST'])
def applyFilters():
    out = None
    with open("./filters/" + request.args.get("filter") + ".json") as f:
        filter = json.load(f)
    raw_data = flask.request.get_data(False, False, False)
    fd, path = tempfile.mkstemp()
    pathout = tempfile.mktemp() + ".mp4"
    try:
        try:
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(raw_data)
                cap = cv2.VideoCapture(path)
                try:
                    while True:
                        success, img = cap.read()
                        if not success:
                            break
                        featuresList = detect_features(
                            img, face_detector, feature_detector)
                        if featuresList is not None:
                            new_img = img
                            for i in range(0, len(featuresList)):
                                features = featuresList[i]
                                new_features = get_new_features(
                                    features, filter)
                                new_img = transform(
                                    new_img, features, new_features)
                                #for x, y in new_features:
                                    #cv2.circle(img=new_img, center=(x, y),
                                               #radius=3, color=(0, 255, 0), thickness=-1)
                                if out is None:
                                    height, width, channels = new_img.shape
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    out = cv2.VideoWriter(
                                        pathout, fourcc, 30, (width, height))
                                out.write(new_img)
                        else:
                            new_img = img
                    out.release()
                    cap.release()

                    pathfinal = tempfile.mktemp() + ".mp4"
                    try:
                        originalVideo = ffmpeg.input(path)
                        warpedVideo = ffmpeg.input(pathout)
                        stream = ffmpeg.output(
                            warpedVideo.video, originalVideo.audio, pathfinal)
                        ffmpeg.run(stream)

                        return send_file(pathfinal, attachment_filename='final.mp4')
                    finally:
                        os.remove(pathfinal)
                finally:
                    print('All done')
        finally:
            os.remove(path)
    finally:
        os.remove(pathout)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")