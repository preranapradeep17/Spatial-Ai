using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

namespace SpatialAI.Agent
{
    [System.Serializable]
    public class QueryRequestPayload
    {
        public string query;
    }

    [System.Serializable]
    public class QueryResponsePayload
    {
        public string query;
        public string final_response;
        public bool completed;
    }

    /// <summary>
    /// Lightweight in-game chat interface to communicate with the Python backend agent.
    /// </summary>
    public class ChatUIController : MonoBehaviour
    {
        [Header("Backend Endpoint")]
        public string agentUrl = "http://localhost:8000/agent/query";
        public UnityActionReceiver actionReceiver;

        private string inputQuery = "Where is the nearest exit?";
        private string chatHistory = "SpatialAI: Ready for spatial commands.\n";
        private Vector2 scrollPos;

        void OnGUI()
        {
            GUI.Box(new Rect(20, 20, 400, 300), "SpatialAI Natural Language Interface");

            GUILayout.BeginArea(new Rect(30, 50, 380, 260));

            scrollPos = GUILayout.BeginScrollView(scrollPos, GUILayout.Height(180));
            GUILayout.Label(chatHistory);
            GUILayout.EndScrollView();

            GUILayout.BeginHorizontal();
            inputQuery = GUILayout.TextField(inputQuery, GUILayout.Width(280));
            if (GUILayout.Button("Send", GUILayout.Width(80)) || (Event.current.isKey && Event.current.keyCode == KeyCode.Return))
            {
                if (!string.IsNullOrEmpty(inputQuery))
                {
                    string q = inputQuery;
                    chatHistory += "User: " + q + "\n";
                    inputQuery = "";
                    StartCoroutine(SendQuery(q));
                }
            }
            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }

        private IEnumerator SendQuery(string query)
        {
            QueryRequestPayload payload = new QueryRequestPayload { query = query };
            string json = JsonUtility.ToJson(payload);

            using (UnityWebRequest req = new UnityWebRequest(agentUrl, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                req.uploadHandler = new UploadHandlerRaw(bodyRaw);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");

                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    QueryResponsePayload res = JsonUtility.FromJson<QueryResponsePayload>(req.downloadHandler.text);
                    chatHistory += "Agent: " + res.final_response + "\n";
                }
                else
                {
                    chatHistory += "Error: Failed to reach backend (" + req.error + ")\n";
                }
            }
        }
    }
}
