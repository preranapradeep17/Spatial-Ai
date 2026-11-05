using UnityEngine;
using System.Collections.Generic;

namespace SpatialAI.Scene
{
    /// <summary>
    /// Metadata component attached to each identifiable laboratory/facility object.
    /// Standardized to the 6 target classes:
    /// 0: table, 1: chair, 2: door, 3: bookshelf, 4: sofa, 5: cabinet
    /// </summary>
    public class LabObject : MonoBehaviour
    {
        public int classId;
        public string className;
        public string uniqueId;
        public Bounds objectBounds;
    }

    /// <summary>
    /// Procedurally constructs a 3D multi-zone investigation facility for Spatial Quest:
    /// - Zone 1: Reception Area (Reception table, visitor sofa, entry door, dataset terminal)
    /// - Zone 2: Investigation Lab (Research workbenches, rolling chairs, steel cabinets, datapad clue)
    /// - Zone 3: Clue Archive (Towering bookshelves, Archive Door 3, electronic wall safe)
    /// </summary>
    public class LabSceneGenerator : MonoBehaviour
    {
        [Header("Facility Dimensions")]
        public float roomWidth = 10.0f;
        public float roomLength = 24.0f; // Multi-zone long facility
        public float roomHeight = 3.5f;

        private GameObject labRoot;
        private List<LabObject> spawnedObjects = new List<LabObject>();

        void Start()
        {
            GenerateFacility();
        }

        [ContextMenu("Regenerate Facility")]
        public void GenerateFacility(int seed = 42)
        {
            Random.InitState(seed);

            if (labRoot != null)
            {
                DestroyImmediate(labRoot);
            }

            labRoot = new GameObject("SpatialQuestFacility");
            spawnedObjects.Clear();

            BuildFacilityArchitecture();
            BuildZone1Reception();
            BuildZone2InvestigationLab();
            BuildZone3ClueArchive();
        }

        private void BuildFacilityArchitecture()
        {
            // Floor
            GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "FacilityFloor";
            floor.transform.parent = labRoot.transform;
            floor.transform.position = new Vector3(0, -0.1f, 0);
            floor.transform.localScale = new Vector3(roomWidth, 0.2f, roomLength);
            SetColor(floor, new Color(0.82f, 0.83f, 0.85f));

            // Ceiling
            GameObject ceiling = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ceiling.name = "FacilityCeiling";
            ceiling.transform.parent = labRoot.transform;
            ceiling.transform.position = new Vector3(0, roomHeight + 0.1f, 0);
            ceiling.transform.localScale = new Vector3(roomWidth, 0.2f, roomLength);
            SetColor(ceiling, new Color(0.92f, 0.92f, 0.94f));

            // Exterior Walls
            CreateWall("Wall_North", new Vector3(0, roomHeight / 2f, roomLength / 2f), new Vector3(roomWidth, roomHeight, 0.2f));
            CreateWall("Wall_South", new Vector3(0, roomHeight / 2f, -roomLength / 2f), new Vector3(roomWidth, roomHeight, 0.2f));
            CreateWall("Wall_East", new Vector3(roomWidth / 2f, roomHeight / 2f, 0), new Vector3(0.2f, roomHeight, roomLength));
            CreateWall("Wall_West", new Vector3(-roomWidth / 2f, roomHeight / 2f, 0), new Vector3(0.2f, roomHeight, roomLength));

            // Zone Partition 1 (Between Reception and Lab at Z = -4)
            CreateWall("Partition_Zone1_2", new Vector3(2.5f, roomHeight / 2f, -4.0f), new Vector3(5.0f, roomHeight, 0.2f));
            CreateWall("Partition_Zone1_2_Left", new Vector3(-3.5f, roomHeight / 2f, -4.0f), new Vector3(3.0f, roomHeight, 0.2f));

            // Zone Partition 2 (Between Lab and Archive at Z = 4)
            CreateWall("Partition_Zone2_3", new Vector3(-2.5f, roomHeight / 2f, 4.0f), new Vector3(5.0f, roomHeight, 0.2f));
            CreateWall("Partition_Zone2_3_Right", new Vector3(3.5f, roomHeight / 2f, 4.0f), new Vector3(3.0f, roomHeight, 0.2f));

            // Overhead Point Lights per Zone
            CreateZoneLight("Light_Zone1", new Vector3(0, roomHeight - 0.3f, -8.0f), new Color(1.0f, 0.95f, 0.9f));
            CreateZoneLight("Light_Zone2", new Vector3(0, roomHeight - 0.3f, 0.0f), new Color(0.9f, 0.95f, 1.0f));
            CreateZoneLight("Light_Zone3", new Vector3(0, roomHeight - 0.3f, 8.0f), new Color(0.95f, 0.9f, 1.0f));
        }

        private void BuildZone1Reception()
        {
            // Reception Desk Table (Class 0)
            CreateTable("table_reception", new Vector3(0f, 0f, -8.0f));

            // Receptionist Chair (Class 1)
            CreateChair("chair_reception", new Vector3(0f, 0f, -7.2f), Quaternion.Euler(0, 180, 0));

            // Visitor Sofa (Class 4)
            CreateSofa("sofa_reception", new Vector3(-3.2f, 0f, -9.0f), Quaternion.Euler(0, 90, 0));

            // Main Entrance Door (Class 2)
            CreateDoor("door_entrance", new Vector3(0f, 0f, -roomLength / 2f + 0.1f));

            // Terminal Station Console (Mission F Station)
            CreateTerminalStation("terminal_reception", new Vector3(3.5f, 0f, -7.5f));
        }

        private void BuildZone2InvestigationLab()
        {
            // Research Workbench Table (Class 0)
            CreateTable("table_lab_center", new Vector3(0f, 0f, 0.0f));

            // Research Chair (Class 1)
            CreateChair("chair_lab_01", new Vector3(-1.2f, 0f, -0.2f), Quaternion.Euler(0, 45, 0));

            // Steel Equipment Storage Cabinet (Class 5)
            CreateCabinet("cabinet_lab_01", new Vector3(3.8f, 0f, 0.5f));

            // Laboratory Chemical Bookshelf (Class 3)
            CreateBookshelf("bookshelf_lab_01", new Vector3(-3.8f, 0f, 1.5f), Quaternion.Euler(0, 90, 0));

            // Interactive Datapad Clue placed on table
            CreateClueDatapad("clue_pda_02", new Vector3(0.2f, 0.76f, 0.1f));
        }

        private void BuildZone3ClueArchive()
        {
            // Towering Archive Bookshelves (Class 3)
            CreateBookshelf("bookshelf_archive_01", new Vector3(-3.5f, 0f, 7.5f), Quaternion.Euler(0, 90, 0));
            CreateBookshelf("bookshelf_archive_02", new Vector3(-3.5f, 0f, 10.0f), Quaternion.Euler(0, 90, 0));

            // Archive Study Table & Chair (Classes 0 & 1)
            CreateTable("table_archive", new Vector3(0.5f, 0f, 8.5f));
            CreateChair("chair_archive", new Vector3(0.5f, 0f, 7.6f), Quaternion.Euler(0, 0, 0));

            // Heavy Secure Cabinet (Class 5)
            CreateCabinet("cabinet_archive_vault", new Vector3(3.8f, 0f, 9.0f));

            // Electronic Wall Safe Clue
            CreateClueWallSafe("clue_safe_01", new Vector3(3.9f, 1.2f, 6.5f));

            // Archive Door 3 (Class 2) on North Partition
            CreateDoor("door_archive_03", new Vector3(0f, 0f, roomLength / 2f - 0.1f));
        }

        // ================= OBJECT FACTORY METHODS =================

        public GameObject CreateTable(string id, Vector3 pos)
        {
            GameObject table = new GameObject(id);
            table.transform.parent = labRoot.transform;
            table.transform.position = pos;

            GameObject top = GameObject.CreatePrimitive(PrimitiveType.Cube);
            top.transform.parent = table.transform;
            top.transform.localPosition = new Vector3(0, 0.73f, 0);
            top.transform.localScale = new Vector3(1.8f, 0.06f, 1.0f);
            SetColor(top, new Color(0.72f, 0.53f, 0.35f));

            BoxCollider col = table.AddComponent<BoxCollider>();
            col.center = new Vector3(0, 0.38f, 0);
            col.size = new Vector3(1.8f, 0.76f, 1.0f);

            LabObject meta = table.AddComponent<LabObject>();
            meta.classId = 0;
            meta.className = "table";
            meta.uniqueId = id;
            spawnedObjects.Add(meta);
            return table;
        }

        public GameObject CreateChair(string id, Vector3 pos, Quaternion rot)
        {
            GameObject chair = new GameObject(id);
            chair.transform.parent = labRoot.transform;
            chair.transform.position = pos;
            chair.transform.rotation = rot;

            GameObject seat = GameObject.CreatePrimitive(PrimitiveType.Cube);
            seat.transform.parent = chair.transform;
            seat.transform.localPosition = new Vector3(0, 0.45f, 0);
            seat.transform.localScale = new Vector3(0.5f, 0.05f, 0.5f);
            SetColor(seat, new Color(0.2f, 0.2f, 0.25f));

            BoxCollider col = chair.AddComponent<BoxCollider>();
            col.center = new Vector3(0, 0.5f, -0.1f);
            col.size = new Vector3(0.55f, 1.0f, 0.55f);

            LabObject meta = chair.AddComponent<LabObject>();
            meta.classId = 1;
            meta.className = "chair";
            meta.uniqueId = id;
            spawnedObjects.Add(meta);
            return chair;
        }

        public GameObject CreateDoor(string id, Vector3 pos)
        {
            GameObject door = new GameObject(id);
            door.transform.parent = labRoot.transform;
            door.transform.position = pos;

            GameObject panel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            panel.transform.parent = door.transform;
            panel.transform.localPosition = new Vector3(0, 1.1f, 0);
            panel.transform.localScale = new Vector3(1.1f, 2.2f, 0.08f);
            SetColor(panel, new Color(0.85f, 0.88f, 0.85f));

            GameObject sign = GameObject.CreatePrimitive(PrimitiveType.Cube);
            sign.transform.parent = door.transform;
            sign.transform.localPosition = new Vector3(0, 2.3f, -0.05f);
            sign.transform.localScale = new Vector3(0.5f, 0.18f, 0.05f);
            SetColor(sign, new Color(0.1f, 0.8f, 0.2f));

            BoxCollider col = door.AddComponent<BoxCollider>();
            col.center = new Vector3(0, 1.15f, 0);
            col.size = new Vector3(1.2f, 2.4f, 0.2f);

            LabObject meta = door.AddComponent<LabObject>();
            meta.classId = 2;
            meta.className = "door";
            meta.uniqueId = id;
            spawnedObjects.Add(meta);
            return door;
        }

        public GameObject CreateBookshelf(string id, Vector3 pos, Quaternion rot)
        {
            GameObject bs = new GameObject(id);
            bs.transform.parent = labRoot.transform;
            bs.transform.position = pos;
            bs.transform.rotation = rot;

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cube);
            body.transform.parent = bs.transform;
            body.transform.localPosition = new Vector3(0, 1.0f, 0);
            body.transform.localScale = new Vector3(1.4f, 2.0f, 0.45f);
            SetColor(body, new Color(0.45f, 0.28f, 0.16f)); // Dark mahogany wood

            BoxCollider col = bs.AddComponent<BoxCollider>();
            col.center = new Vector3(0, 1.0f, 0);
            col.size = new Vector3(1.45f, 2.05f, 0.5f);

            LabObject meta = bs.AddComponent<LabObject>();
            meta.classId = 3;
            meta.className = "bookshelf";
            meta.uniqueId = id;
            spawnedObjects.Add(meta);
            return bs;
        }

        public GameObject CreateSofa(string id, Vector3 pos, Quaternion rot)
        {
            GameObject sofa = new GameObject(id);
            sofa.transform.parent = labRoot.transform;
            sofa.transform.position = pos;
            sofa.transform.rotation = rot;

            GameObject baseCushion = GameObject.CreatePrimitive(PrimitiveType.Cube);
            baseCushion.transform.parent = sofa.transform;
            baseCushion.transform.localPosition = new Vector3(0, 0.35f, 0);
            baseCushion.transform.localScale = new Vector3(1.8f, 0.4f, 0.85f);
            SetColor(baseCushion, new Color(0.18f, 0.25f, 0.35f)); // Navy leather

            BoxCollider col = sofa.AddComponent<BoxCollider>();
            col.center = new Vector3(0, 0.45f, 0);
            col.size = new Vector3(1.9f, 0.85f, 0.9f);

            LabObject meta = sofa.AddComponent<LabObject>();
            meta.classId = 4;
            meta.className = "sofa";
            meta.uniqueId = id;
            spawnedObjects.Add(meta);
            return sofa;
        }

        public GameObject CreateCabinet(string id, Vector3 pos)
        {
            GameObject cab = new GameObject(id);
            cab.transform.parent = labRoot.transform;
            cab.transform.position = pos;

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cube);
            body.transform.parent = cab.transform;
            body.transform.localPosition = new Vector3(0, 0.85f, 0);
            body.transform.localScale = new Vector3(0.9f, 1.7f, 0.55f);
            SetColor(body, new Color(0.6f, 0.65f, 0.7f)); // Industrial steel gray

            BoxCollider col = cab.AddComponent<BoxCollider>();
            col.center = new Vector3(0, 0.85f, 0);
            col.size = new Vector3(0.95f, 1.75f, 0.6f);

            LabObject meta = cab.AddComponent<LabObject>();
            meta.classId = 5;
            meta.className = "cabinet";
            meta.uniqueId = id;
            spawnedObjects.Add(meta);
            return cab;
        }

        public GameObject CreateTerminalStation(string id, Vector3 pos)
        {
            GameObject term = new GameObject(id);
            term.transform.parent = labRoot.transform;
            term.transform.position = pos;

            GameObject pedestal = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pedestal.transform.parent = term.transform;
            pedestal.transform.localPosition = new Vector3(0, 0.55f, 0);
            pedestal.transform.localScale = new Vector3(0.6f, 1.1f, 0.4f);
            SetColor(pedestal, new Color(0.25f, 0.28f, 0.32f));

            GameObject screen = GameObject.CreatePrimitive(PrimitiveType.Cube);
            screen.transform.parent = term.transform;
            screen.transform.localPosition = new Vector3(0, 1.15f, 0.1f);
            screen.transform.localRotation = Quaternion.Euler(-25, 0, 0);
            screen.transform.localScale = new Vector3(0.5f, 0.35f, 0.05f);
            SetColor(screen, new Color(0.05f, 0.7f, 0.9f)); // Glowing cyan UI screen

            return term;
        }

        public GameObject CreateClueDatapad(string id, Vector3 pos)
        {
            GameObject pda = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pda.name = id;
            pda.transform.parent = labRoot.transform;
            pda.transform.position = pos;
            pda.transform.localScale = new Vector3(0.22f, 0.02f, 0.32f);
            SetColor(pda, new Color(0.1f, 0.15f, 0.2f));
            return pda;
        }

        public GameObject CreateClueWallSafe(string id, Vector3 pos)
        {
            GameObject safe = GameObject.CreatePrimitive(PrimitiveType.Cube);
            safe.name = id;
            safe.transform.parent = labRoot.transform;
            safe.transform.position = pos;
            safe.transform.localScale = new Vector3(0.45f, 0.45f, 0.25f);
            SetColor(safe, new Color(0.12f, 0.12f, 0.15f));
            return safe;
        }

        private void CreateWall(string name, Vector3 pos, Vector3 scale)
        {
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = name;
            wall.transform.parent = labRoot.transform;
            wall.transform.position = pos;
            wall.transform.localScale = scale;
            SetColor(wall, new Color(0.9f, 0.92f, 0.94f));
        }

        private void CreateZoneLight(string name, Vector3 pos, Color c)
        {
            GameObject lightObj = new GameObject(name);
            lightObj.transform.parent = labRoot.transform;
            lightObj.transform.position = pos;
            Light l = lightObj.AddComponent<Light>();
            l.type = LightType.Point;
            l.range = 14.0f;
            l.intensity = 1.1f;
            l.color = c;
        }

        private void SetColor(GameObject obj, Color color)
        {
            Renderer r = obj.GetComponent<Renderer>();
            if (r != null)
            {
                r.material = new Material(Shader.Find("Standard"));
                r.material.color = color;
            }
        }
    }
}
