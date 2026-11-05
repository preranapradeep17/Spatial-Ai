using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

namespace SpatialAI.Perception
{
    /// <summary>
    /// Captures live Unity camera frames and submits them to the FastAPI /perception endpoint.
    /// </summary>
    public class RuntimePerceptionSender : MonoBehaviour
    {
        [Header("Backend Connection")]
        public string backendUrl = "http://localhost:8000/perception";
        public float captureIntervalSeconds = 1.0f;
        public bool sendMockRequestForTesting = false;

        private Camera perceptionCamera;
        private RenderTexture rt;
        private Texture2D tex;
        private bool isSending = false;

        void Start()
        {
            perceptionCamera = GetComponent<Camera>();
            if (perceptionCamera == null)
            {
                perceptionCamera = Camera.main;
            }

            rt = new RenderTexture(640, 480, 24, RenderTextureFormat.ARGB32);
            tex = new Texture2D(640, 480, TextureFormat.RGB24, false);

            StartCoroutine(PerceptionLoop());
        }

        private IEnumerator PerceptionLoop()
        {
            while (true)
            {
                yield return new WaitForSeconds(captureIntervalSeconds);
                if (!isSending)
                {
                    StartCoroutine(SendPerceptionPayload());
                }
            }
        }

        private IEnumerator SendPerceptionPayload()
        {
            isSending = true;

            string jsonPayload;

            if (sendMockRequestForTesting)
            {
                jsonPayload = "{\"mock_mode\": true}";
            }
            else
            {
                perceptionCamera.targetTexture = rt;
                perceptionCamera.Render();
                RenderTexture.active = rt;
                tex.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
                tex.Apply();
                perceptionCamera.targetTexture = null;
                RenderTexture.active = null;

                byte[] pngBytes = tex.EncodeToPNG();
                string base64Image = System.Convert.ToBase64String(pngBytes);

                jsonPayload = $"{{\"image_base64\": \"{base64Image}\", \"mock_mode\": false}}";
            }

            using (UnityWebRequest req = new UnityWebRequest(backendUrl, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                req.uploadHandler = new UploadHandlerRaw(bodyRaw);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");

                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    Debug.Log("[SpatialAI] Perception Response: " + req.downloadHandler.text);
                }
                else
                {
                    Debug.LogWarning("[SpatialAI] Perception request failed: " + req.error);
                }
            }

            isSending = false;
        }

        void OnDestroy()
        {
            if (rt != null) Destroy(rt);
            if (tex != null) Destroy(tex);
        }
    }
}
