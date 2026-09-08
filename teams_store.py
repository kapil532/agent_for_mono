"""
Simple JSON-file storage for teams and members.
File: ~/tiny_llm/teams.json
"""
import os
import json

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teams.json')


def _load():
    if not os.path.exists(STORE_PATH):
        return {"teams": []}
    with open(STORE_PATH) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"teams": []}


def _save(data):
    with open(STORE_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def list_teams():
    return _load()["teams"]


def get_team(name):
    for t in _load()["teams"]:
        if t["name"].lower() == name.lower():
            return t
    return None


def create_team(name):
    data = _load()
    name = name.strip()
    if not name:
        return {"error": "Team name is required"}
    if any(t["name"].lower() == name.lower() for t in data["teams"]):
        return {"error": f"Team '{name}' already exists"}
    team = {"name": name, "members": []}
    data["teams"].append(team)
    _save(data)
    return team


def delete_team(name):
    data = _load()
    data["teams"] = [t for t in data["teams"] if t["name"].lower() != name.lower()]
    _save(data)
    return {"ok": True}


def add_member(team_name, member):
    data = _load()
    for t in data["teams"]:
        if t["name"].lower() == team_name.lower():
            if any(m["accountId"] == member["accountId"] for m in t["members"]):
                return {"error": f"{member['displayName']} is already in this team"}
            t["members"].append({
                "accountId": member["accountId"],
                "displayName": member["displayName"],
                "email": member.get("email", ""),
                "avatar": member.get("avatar", ""),
            })
            _save(data)
            return t
    return {"error": f"Team '{team_name}' not found"}


def remove_member(team_name, account_id):
    data = _load()
    for t in data["teams"]:
        if t["name"].lower() == team_name.lower():
            t["members"] = [m for m in t["members"] if m["accountId"] != account_id]
            _save(data)
            return t
    return {"error": f"Team '{team_name}' not found"}
