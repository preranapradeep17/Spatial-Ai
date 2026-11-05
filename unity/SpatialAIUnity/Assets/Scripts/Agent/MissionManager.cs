using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

namespace SpatialAI.Agent
{
    public enum MissionType
    {
        MissionA_VisionScout,       // "What objects are visible?"
        MissionB_SpatialDetective,  // "What is left of table?", "Calculate distance"
        MissionC_FindAndNavigate,   // "Guide me to nearest door"
        MissionD_ClueRoom,          // Document retrieval & clue inspection
        MissionE_TimedInvestigation,// Sequential timed multi-objective puzzle
        MissionF_DatasetTerminal    // Inspect real SUN RGB-D dataset samples
    }

    [System.Serializable]
    public class MissionObjective
    {
        public string objectiveId;
        public string description;
        public bool isCompleted;
        public int points;
    }

    /// <summary>
    /// Mission and Puzzle Manager for Spatial Quest.
    /// Tracks active mission objectives, scores, investigation timer, and clue states.
    /// </summary>
    public class MissionManager : MonoBehaviour
    {
        [Header("Active Mission Settings")]
        public MissionType activeMission = MissionType.MissionA_VisionScout;
        public float missionTimer = 180.0f; // 3 minutes for timed investigation
        public bool isTimerRunning = false;
        public int currentScore = 0;

        [Header("Mission Objectives")]
        public List<MissionObjective> currentObjectives = new List<MissionObjective>();

        [Header("Investigation State")]
        public bool door3Unlocked = false;
        public bool safeUnlocked = false;
        public bool ventilationActive = false;
        public string discoveredOverrideCode = "";

        // Events
        public event Action<int, string> OnScoreChanged;
        public event Action<string> OnObjectiveCompleted;
        public event Action<string> OnMissionCompleted;

        void Start()
        {
            InitializeMission(activeMission);
        }

        void Update()
        {
            if (isTimerRunning && missionTimer > 0)
            {
                missionTimer -= Time.deltaTime;
                if (missionTimer <= 0)
                {
                    missionTimer = 0;
                    isTimerRunning = false;
                    Debug.LogWarning("[MissionManager] Timed investigation period expired!");
                }
            }
        }

        public void InitializeMission(MissionType mission)
        {
            activeMission = mission;
            currentObjectives.Clear();

            switch (mission)
            {
                case MissionType.MissionA_VisionScout:
                    currentObjectives.Add(new MissionObjective { objectiveId = "scout_01", description = "Query companion drone: 'What objects are visible?'", points = 100 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "scout_02", description = "Locate table in Zone 1", points = 100 });
                    break;

                case MissionType.MissionB_SpatialDetective:
                    currentObjectives.Add(new MissionObjective { objectiveId = "spatial_01", description = "Identify what object is left of the research table", points = 150 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "spatial_02", description = "Calculate Euclidean distance between chair and table", points = 150 });
                    break;

                case MissionType.MissionC_FindAndNavigate:
                    currentObjectives.Add(new MissionObjective { objectiveId = "nav_01", description = "Find closest door using spatial perception", points = 150 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "nav_02", description = "Command companion drone to navigate to door", points = 200 });
                    break;

                case MissionType.MissionD_ClueRoom:
                    currentObjectives.Add(new MissionObjective { objectiveId = "clue_01", description = "Query RAG system for Door 3 override passcode", points = 200 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "clue_02", description = "Inspect clue_safe_01 in Zone 3 Archive", points = 200 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "clue_03", description = "Interact with Door 3 to unlock containment", points = 250 });
                    break;

                case MissionType.MissionE_TimedInvestigation:
                    isTimerRunning = true;
                    missionTimer = 180.0f;
                    currentObjectives.Add(new MissionObjective { objectiveId = "timed_01", description = "Detect all objects in Investigation Lab", points = 100 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "timed_02", description = "Retrieve override passcode via RAG", points = 150 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "timed_03", description = "Unlock Archive Door 3 before timer expires", points = 300 });
                    break;

                case MissionType.MissionF_DatasetTerminal:
                    currentObjectives.Add(new MissionObjective { objectiveId = "term_01", description = "Access Terminal Station in Reception", points = 100 });
                    currentObjectives.Add(new MissionObjective { objectiveId = "term_02", description = "Inspect SUN RGB-D benchmark sample predictions", points = 200 });
                    break;
            }

            Debug.Log($"[MissionManager] Initialized {activeMission} with {currentObjectives.Count} objectives.");
        }

        public void CompleteObjective(string objectiveId)
        {
            foreach (var obj in currentObjectives)
            {
                if (obj.objectiveId == objectiveId && !obj.isCompleted)
                {
                    obj.isCompleted = true;
                    currentScore += obj.points;
                    OnScoreChanged?.Invoke(currentScore, obj.description);
                    OnObjectiveCompleted?.Invoke(objectiveId);
                    Debug.Log($"[MissionManager] Objective Completed: {obj.description} (+{obj.points} pts. Total: {currentScore})");
                    break;
                }
            }

            // Check if all completed
            bool allDone = true;
            foreach (var obj in currentObjectives)
            {
                if (!obj.isCompleted)
                {
                    allDone = false;
                    break;
                }
            }

            if (allDone)
            {
                Debug.Log($"[MissionManager] Mission Completed: {activeMission}!");
                OnMissionCompleted?.Invoke(activeMission.ToString());
            }
        }
    }
}
