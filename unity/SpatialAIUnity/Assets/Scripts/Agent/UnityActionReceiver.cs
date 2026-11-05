using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

namespace SpatialAI.Agent
{
    [System.Serializable]
    public class ActionCallbackPayload
    {
        public string action_id;
        public string action_type;
        public bool success;
        public string message;
        public float[] avatar_position;
    }

    /// <summary>
    /// Receives agent commands (e.g. navigate_to, highlight_object) and executes safe physical avatar navigation.
    /// </summary>
    public class UnityActionReceiver : MonoBehaviour
    {
        [Header("Backend Callback")]
        public string callbackUrl = "http://localhost:8000/unity/action-result";
        public float moveSpeed = 3.0f;

        private Vector3 targetDestination;
        private bool isMoving = false;
        private string currentActionId;

        public void ExecuteNavigateTo(string actionId, Vector3 destination)
        {
            currentActionId = actionId;
            targetDestination = new Vector3(destination.x, transform.position.y, destination.z);
            isMoving = true;
            Debug.Log($"[SpatialAI] Navigating to target: {targetDestination}");
        }

        void Update()
        {
            if (isMoving)
            {
                transform.position = Vector3.MoveTowards(transform.position, targetDestination, moveSpeed * Time.deltaTime);
                transform.LookAt(targetDestination);

                if (Vector3.Distance(transform.position, targetDestination) < 0.2f)
                {
                    isMoving = false;
                    Debug.Log("[SpatialAI] Arrived at destination.");
                    StartCoroutine(ReportActionResult(currentActionId, true, "Navigation completed successfully."));
                }
            }
        }

        private IEnumerator ReportActionResult(string actionId, bool success, string msg)
        {
            ActionCallbackPayload payload = new ActionCallbackPayload
            {
                action_id = actionId ?? System.Guid.NewGuid().ToString(),
                action_type = "navigate_to",
                success = success,
                message = msg,
                avatar_position = new float[] { transform.position.x, transform.position.y, transform.position.z }
            };

            string json = JsonUtility.ToJson(payload);

            using (UnityWebRequest req = new UnityWebRequest(callbackUrl, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                req.uploadHandler = new UploadHandlerRaw(bodyRaw);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");

                yield return req.SendWebRequest();
            }
        }
    }
}
