# Spatial Quest — Mission Walkthrough & Puzzle Specification

## Overview
Spatial Quest challenges the player to solve investigative mysteries across 6 distinct missions (A through F), demonstrating the full suite of Computer Vision, Spatial Localization, Custom Transformer Reasoning, and Local RAG.

---

## Mission A: Vision Scout
- **Location**: Zone 1 (Reception Checkpoint)
- **Objective**: Survey the visual environment and verify object recognition.
- **Player Dialogue Example**:
  > *Player*: "What objects are visible around us?"  
  > *Assistant*: `<thought>I should inspect the environment to identify all detected objects.</thought><tool_call>{"name": "get_visible_objects", "arguments": {}}</tool_call>`  
  > *Observation*: `[{"class_name": "table", "position": [0.0, 0.0, -8.0]}, {"class_name": "sofa", "position": [-3.2, 0.0, -9.0]}]`  
  > *Assistant*: "I have detected a reception table directly ahead and a visitor sofa 3.2 meters to your left."
- **Award**: +200 Points

---

## Mission B: Spatial Detective
- **Location**: Zone 2 (Investigation Laboratory)
- **Objective**: Use 3D coordinate geometry to verify evidence proximity.
- **Player Dialogue Example**:
  > *Player*: "What is to the left of the central research table?"  
  > *Assistant*: `<tool_call>{"name": "query_spatial_relation", "arguments": {"object_name": "table", "relation": "left_of"}}</tool_call>`  
  > *Observation*: `{"matches": ["chair_lab_01"]}`  
  > *Assistant*: "The research chair is positioned to the left of the table."  
  > *Player*: "How far is the chair from the table?"  
  > *Assistant*: `<tool_call>{"name": "calculate_distance", "arguments": {"object_a": "chair", "object_b": "table"}}</tool_call>`  
  > *Assistant*: "The physical Euclidean distance is 1.02 meters."
- **Award**: +300 Points

---

## Mission C: Find and Navigate
- **Location**: Zone 2 to Zone 3 Corridor
- **Objective**: Locate the locked doorway and guide the companion drone.
- **Player Dialogue Example**:
  > *Player*: "Find the nearest door and calculate a path."  
  > *Assistant*: `<tool_call>{"name": "find_nearest_target", "arguments": {"target_class": "door"}}</tool_call>`  
  > *Observation*: `{"found": true, "position": [0.0, 0.0, 11.9], "distance": 11.9}`  
  > *Assistant*: `<tool_call>{"name": "find_path", "arguments": {"destination": "door"}}</tool_call>`  
  > *Assistant*: "Clear path computed across 3 waypoints avoiding the central workbenches."
- **Award**: +350 Points

---

## Mission D: Clue Room & RAG Investigation
- **Location**: Zone 3 (Clue Archive & Vault)
- **Objective**: Retrieve the emergency override passcode from facility incident logs and unlock Door 3.
- **Player Dialogue Example**:
  > *Player*: "What is the emergency override code for Door 3?"  
  > *Assistant*: `<tool_call>{"name": "search_documents", "arguments": {"query": "Door 3 manual override code"}}</tool_call>`  
  > *Observation*: `[{"title": "Maintenance Log", "text": "Notice: Keypad on Door 3 was reset to default emergency maintenance code 7391."}]`  
  > *Assistant*: "According to the facility maintenance log, the emergency override code for Door 3 is **7391**."  
  > *Player*: "Unlock the door."  
  > *Assistant*: `<tool_call>{"name": "interact_with_object", "arguments": {"object_id": "door"}}</tool_call>`  
  > *Assistant*: "Door 3 magnetic lockdown disengaged. Vault corridor is open!"
- **Award**: +650 Points

---

## Mission E: Timed Investigation
- **Location**: Whole Facility (Sector 4)
- **Objective**: Complete all investigative objectives before the 180-second containment timer expires.
- **Reward**: High-score multiplier based on remaining time.

---

## Mission F: Real Dataset Terminal
- **Location**: Terminal Station Console in Zone 1 (Reception)
- **Objective**: Academic interactive viva station. The player or evaluator uses the in-game terminal to inspect the real SUN RGB-D benchmark dataset samples, depth decoding bitshift masks, and real 3D bounding box predictions.
- **Award**: Viva Demonstration Readiness Certification.
