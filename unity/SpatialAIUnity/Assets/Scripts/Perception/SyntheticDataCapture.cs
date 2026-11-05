using UnityEngine;
using System.IO;
using System.Collections.Generic;
using SpatialAI.Scene;

namespace SpatialAI.Perception
{
    [System.Serializable]
    public class BBox2DData
    {
        public float xmin;
        public float ymin;
        public float xmax;
        public float ymax;
    }

    [System.Serializable]
    public class AnnotationObjectData
    {
        public string object_id;
        public int class_id;
        public string class_name;
        public BBox2DData bbox_2d;
        public float[] position_world_3d;
        public float[] position_camera_3d;
        public float distance_to_camera;
        public bool is_visible;
    }

    [System.Serializable]
    public class CameraMetadata
    {
        public float fx;
        public float fy;
        public float cx;
        public float cy;
        public int image_width;
        public int image_height;
        public float[] position;
        public float[] rotation_quaternion;
    }

    [System.Serializable]
    public class FrameAnnotation
    {
        public string frame_id;
        public string image_filename;
        public string depth_filename;
        public CameraMetadata camera;
        public List<AnnotationObjectData> objects = new List<AnnotationObjectData>();
    }

    /// <summary>
    /// Captures RGB images, depth maps, and ground-truth bounding box annotations from Unity.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class SyntheticDataCapture : MonoBehaviour
    {
        [Header("Capture Settings")]
        public int captureWidth = 640;
        public int captureHeight = 480;
        public string outputDirectory = "DatasetExport";
        public int numberOfFramesToCapture = 10;

        private Camera targetCamera;
        private RenderTexture rgbRenderTexture;
        private Texture2D rgbTexture2D;

        void Awake()
        {
            targetCamera = GetComponent<Camera>();
            targetCamera.depthTextureMode = DepthTextureMode.Depth;
        }

        [ContextMenu("Execute Capture Sequence")]
        public void CaptureSequence()
        {
            string fullOutDir = Path.Combine(Application.dataPath, "..", outputDirectory);
            string rgbDir = Path.Combine(fullOutDir, "rgb");
            string depthDir = Path.Combine(fullOutDir, "depth");
            string annotDir = Path.Combine(fullOutDir, "annotations");

            Directory.CreateDirectory(rgbDir);
            Directory.CreateDirectory(depthDir);
            Directory.CreateDirectory(annotDir);

            rgbRenderTexture = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
            rgbTexture2D = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

            targetCamera.targetTexture = rgbRenderTexture;

            LabObject[] allObjects = FindObjectsOfType<LabObject>();

            for (int i = 0; i < numberOfFramesToCapture; i++)
            {
                // Randomize camera viewpoint slightly around the lab center
                float angle = (i / (float)numberOfFramesToCapture) * Mathf.PI * 2f;
                float radius = Random.Range(3.0f, 5.0f);
                float camX = Mathf.Sin(angle) * radius;
                float camZ = Mathf.Cos(angle) * radius;
                float camY = Random.Range(1.2f, 2.0f);

                transform.position = new Vector3(camX, camY, camZ);
                transform.LookAt(new Vector3(0, 0.8f, 2.5f)); // Look toward central lab table

                // Render frame
                targetCamera.Render();
                RenderTexture.active = rgbRenderTexture;
                rgbTexture2D.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
                rgbTexture2D.Apply();

                string frameId = string.Format("frame_{0:D5}", i);
                string rgbPath = Path.Combine(rgbDir, frameId + ".png");
                string depthPath = Path.Combine(depthDir, frameId + ".depth");
                string annotPath = Path.Combine(annotDir, frameId + ".json");

                // Save RGB PNG
                byte[] rgbBytes = rgbTexture2D.EncodeToPNG();
                File.WriteAllBytes(rgbPath, rgbBytes);

                // Compute camera intrinsics
                float fovRad = targetCamera.fieldOfView * Mathf.Deg2Rad;
                float fy = (captureHeight / 2.0f) / Mathf.Tan(fovRad / 2.0f);
                float fx = fy * targetCamera.aspect;
                float cx = captureWidth / 2.0f;
                float cy = captureHeight / 2.0f;

                FrameAnnotation annot = new FrameAnnotation();
                annot.frame_id = frameId;
                annot.image_filename = "rgb/" + frameId + ".png";
                annot.depth_filename = "depth/" + frameId + ".depth";
                annot.camera = new CameraMetadata
                {
                    fx = fx,
                    fy = fy,
                    cx = cx,
                    cy = cy,
                    image_width = captureWidth,
                    image_height = captureHeight,
                    position = new float[] { transform.position.x, transform.position.y, transform.position.z },
                    rotation_quaternion = new float[] { transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w }
                };

                // Annotate visible objects
                foreach (var obj in allObjects)
                {
                    Collider col = obj.GetComponent<Collider>();
                    if (col == null) continue;

                    Bounds b = col.bounds;
                    Vector3[] corners = GetBoundsCorners(b);

                    float minX = float.MaxValue;
                    float maxX = float.MinValue;
                    float minY = float.MaxValue;
                    float maxY = float.MinValue;
                    bool inFront = false;

                    foreach (var c in corners)
                    {
                        Vector3 screenPt = targetCamera.WorldToScreenPoint(c);
                        if (screenPt.z > 0)
                        {
                            inFront = true;
                            if (screenPt.x < minX) minX = screenPt.x;
                            if (screenPt.x > maxX) maxX = screenPt.x;
                            if (screenPt.y < minY) minY = screenPt.y;
                            if (screenPt.y > maxY) maxY = screenPt.y;
                        }
                    }

                    if (!inFront) continue;

                    // Convert screen points (origin bottom-left) to image points (origin top-left)
                    float xmin = Mathf.Clamp(minX, 0, captureWidth);
                    float xmax = Mathf.Clamp(maxX, 0, captureWidth);
                    float ymin = Mathf.Clamp(captureHeight - maxY, 0, captureHeight);
                    float ymax = Mathf.Clamp(captureHeight - minY, 0, captureHeight);

                    // Skip negligible or out-of-frame boxes
                    if ((xmax - xmin) < 5 || (ymax - ymin) < 5) continue;

                    Vector3 camLocalPos = transform.InverseTransformPoint(obj.transform.position);

                    AnnotationObjectData objData = new AnnotationObjectData
                    {
                        object_id = obj.uniqueId,
                        class_id = obj.classId,
                        class_name = obj.className,
                        bbox_2d = new BBox2DData { xmin = xmin, ymin = ymin, xmax = xmax, ymax = ymax },
                        position_world_3d = new float[] { obj.transform.position.x, obj.transform.position.y, obj.transform.position.z },
                        position_camera_3d = new float[] { camLocalPos.x, camLocalPos.y, camLocalPos.z },
                        distance_to_camera = Vector3.Distance(transform.position, obj.transform.position),
                        is_visible = true
                    };
                    annot.objects.Add(objData);
                }

                // Write JSON annotation
                string json = JsonUtility.ToJson(annot, true);
                File.WriteAllText(annotPath, json);
            }

            targetCamera.targetTexture = null;
            RenderTexture.active = null;
            DestroyImmediate(rgbRenderTexture);
            DestroyImmediate(rgbTexture2D);

            Debug.Log($"[SpatialAI] Captured {numberOfFramesToCapture} frames to {fullOutDir}");
        }

        private Vector3[] GetBoundsCorners(Bounds b)
        {
            Vector3 min = b.min;
            Vector3 max = b.max;
            return new Vector3[] {
                new Vector3(min.x, min.y, min.z),
                new Vector3(min.x, min.y, max.z),
                new Vector3(min.x, max.y, min.z),
                new Vector3(min.x, max.y, max.z),
                new Vector3(max.x, min.y, min.z),
                new Vector3(max.x, min.y, max.z),
                new Vector3(max.x, max.y, min.z),
                new Vector3(max.x, max.y, max.z)
            };
        }
    }
}
