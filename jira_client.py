"""
JIRA Client (Atlassian Cloud)
Uses basic auth: email + API token
Get token: https://id.atlassian.com/manage-profile/security/api-tokens
"""
import os
import base64
import requests

# Auto-load .env file (agar hai)
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            v = v.strip().strip('"').strip("'")
            if v and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v

JIRA_DOMAIN = os.environ.get('JIRA_DOMAIN', '')
JIRA_EMAIL = os.environ.get('JIRA_EMAIL', '')
JIRA_TOKEN = os.environ.get('JIRA_TOKEN', '')


def reload_config():
    """Re-read .env after settings save"""
    global JIRA_DOMAIN, JIRA_EMAIL, JIRA_TOKEN
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    JIRA_DOMAIN = os.environ.get('JIRA_DOMAIN', '')
    JIRA_EMAIL = os.environ.get('JIRA_EMAIL', '')
    JIRA_TOKEN = os.environ.get('JIRA_TOKEN', '')


def save_config(domain, email, token):
    """Write credentials to .env file"""
    content = f"""# ═══════════════════════════════════════════════════════
# JIRA CREDENTIALS (saved from Settings UI)
# ═══════════════════════════════════════════════════════

JIRA_DOMAIN={domain}
JIRA_EMAIL={email}
JIRA_TOKEN={token}
"""
    with open(ENV_FILE, 'w') as f:
        f.write(content)
    os.chmod(ENV_FILE, 0o600)
    reload_config()


def get_config_status():
    """Return sanitized config for UI (mask token)"""
    return {
        "domain": JIRA_DOMAIN,
        "email": JIRA_EMAIL,
        "token_set": bool(JIRA_TOKEN),
        "token_hint": (JIRA_TOKEN[:6] + "..." + JIRA_TOKEN[-4:]) if len(JIRA_TOKEN) > 10 else "",
        "configured": is_configured(),
    }


def test_connection():
    """Ping /myself to verify credentials work"""
    headers = _auth_header()
    if not headers:
        return {"ok": False, "error": "Credentials missing"}
    try:
        r = requests.get(f"https://{JIRA_DOMAIN}/rest/api/3/myself", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {"ok": True, "name": data.get("displayName"), "accountId": data.get("accountId")}
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:150]}"}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e)}


def _auth_header():
    email = os.environ.get('JIRA_EMAIL', '')
    token = os.environ.get('JIRA_TOKEN', '')
    if not (email and token):
        return None
    creds = f"{email}:{token}"
    b64 = base64.b64encode(creds.encode()).decode()
    return {
        "Authorization": f"Basic {b64}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def is_configured():
    return bool(os.environ.get('JIRA_EMAIL') and os.environ.get('JIRA_TOKEN'))


def search_issues(jql, max_results=15):
    """Fetch issues from JIRA using JQL"""
    headers = _auth_header()
    if not headers:
        return {"error": "JIRA credentials missing. Set JIRA_EMAIL and JIRA_TOKEN env vars."}

    domain = os.environ.get('JIRA_DOMAIN', '')
    url = f"https://{domain}/rest/api/3/search/jql"
    params = {
        "jql": jql,
        "maxResults": max_results,
        "fields": "summary,status,assignee,priority,issuetype,updated",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            return {"error": f"JIRA API {r.status_code}: {r.text[:200]}"}
        data = r.json()
        issues = []
        for issue in data.get("issues", []):
            f = issue.get("fields", {})
            issues.append({
                "key": issue.get("key"),
                "summary": f.get("summary", ""),
                "status": (f.get("status") or {}).get("name", "-"),
                "assignee": ((f.get("assignee") or {}).get("displayName") or "Unassigned"),
                "priority": (f.get("priority") or {}).get("name", "-"),
                "type": (f.get("issuetype") or {}).get("name", "-"),
                "updated": (f.get("updated") or "")[:10],
                "url": f"https://{domain}/browse/{issue.get('key')}",
            })
        return {"issues": issues, "total": len(issues)}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


import re

JIRA_KEY_REGEX = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')


def extract_jira_keys(text):
    """Find all JIRA keys like MREC-123, A20M-2797 in free text"""
    return list(dict.fromkeys(JIRA_KEY_REGEX.findall(text.upper())))


def _adf_to_text(node, out=None):
    """Convert Atlassian Document Format (ADF) JSON to plain text"""
    if out is None:
        out = []
    if not node:
        return ""
    if isinstance(node, dict):
        if node.get("type") == "text":
            out.append(node.get("text", ""))
        for child in node.get("content", []) or []:
            _adf_to_text(child, out)
        if node.get("type") in ("paragraph", "heading", "listItem", "codeBlock"):
            out.append("\n")
    elif isinstance(node, list):
        for c in node:
            _adf_to_text(c, out)
    return "".join(out).strip()


def get_issue_detail(key):
    """Fetch single issue with description, comments, subtasks"""
    headers = _auth_header()
    if not headers:
        return {"error": "JIRA credentials missing"}
    domain = os.environ.get('JIRA_DOMAIN', '')
    try:
        r = requests.get(
            f"https://{domain}/rest/api/3/issue/{key}",
            headers=headers,
            params={"fields": "summary,status,assignee,reporter,priority,issuetype,updated,created,description,comment,subtasks,labels,duedate,resolution"},
            timeout=15,
        )
        if r.status_code == 404:
            return {"error": f"{key} nahi mila (ya tumhe access nahi hai)"}
        if r.status_code != 200:
            return {"error": f"JIRA API {r.status_code}: {r.text[:200]}"}
        data = r.json()
        f = data.get("fields", {})
        comments = []
        for c in (f.get("comment") or {}).get("comments", [])[-5:]:
            comments.append({
                "author": (c.get("author") or {}).get("displayName", "Unknown"),
                "date": (c.get("created") or "")[:10],
                "body": _adf_to_text(c.get("body")),
            })
        subtasks = []
        for st in f.get("subtasks", []):
            sf = st.get("fields", {})
            subtasks.append({
                "key": st.get("key"),
                "summary": sf.get("summary", ""),
                "status": (sf.get("status") or {}).get("name", "-"),
                "url": f"https://{domain}/browse/{st.get('key')}",
            })
        return {
            "key": data.get("key"),
            "summary": f.get("summary", ""),
            "status": (f.get("status") or {}).get("name", "-"),
            "assignee": ((f.get("assignee") or {}).get("displayName") or "Unassigned"),
            "reporter": ((f.get("reporter") or {}).get("displayName") or "-"),
            "priority": (f.get("priority") or {}).get("name", "-"),
            "type": (f.get("issuetype") or {}).get("name", "-"),
            "labels": f.get("labels", []),
            "created": (f.get("created") or "")[:10],
            "updated": (f.get("updated") or "")[:10],
            "duedate": f.get("duedate") or "-",
            "resolution": ((f.get("resolution") or {}).get("name") or "Unresolved"),
            "description": _adf_to_text(f.get("description")) or "(no description)",
            "comments": comments,
            "subtasks": subtasks,
            "url": f"https://{domain}/browse/{data.get('key')}",
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


def search_users(query, max_results=5):
    """Find JIRA users by name/email"""
    headers = _auth_header()
    if not headers:
        return []
    domain = os.environ.get('JIRA_DOMAIN', '')
    try:
        r = requests.get(
            f"https://{domain}/rest/api/3/user/search",
            headers=headers,
            params={"query": query, "maxResults": max_results},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        return [{
            "accountId": u.get("accountId"),
            "displayName": u.get("displayName"),
            "email": u.get("emailAddress", ""),
            "avatar": (u.get("avatarUrls") or {}).get("48x48", ""),
        } for u in r.json() if u.get("active", True) and u.get("accountId")]
    except requests.exceptions.RequestException:
        return []


def get_tasks_by_user(name, max_results=25):
    """Find user then fetch their assigned tasks"""
    users = search_users(name, max_results=5)
    if not users:
        return {"error": f"No active JIRA user found matching '{name}'"}

    user = users[0]
    jql = f'assignee = "{user["accountId"]}" ORDER BY updated DESC'
    result = search_issues(jql, max_results=max_results)
    if "error" in result:
        return result
    result["user"] = user
    result["other_matches"] = users[1:] if len(users) > 1 else []
    return result


PRODUCTION_FILTER_ID = os.environ.get('JIRA_PRODUCTION_FILTER_ID', '')
PRODUCTION_DASHBOARD_URL = os.environ.get('JIRA_PRODUCTION_DASHBOARD_URL', '')


def get_production_issues(max_results=50):
    """Fetch production issues from the dashboard filter"""
    headers = _auth_header()
    if not headers:
        return {"error": "JIRA credentials missing"}
    domain = os.environ.get('JIRA_DOMAIN', '')

    # First, get the filter's JQL (in case it changes on JIRA side)
    try:
        fr = requests.get(
            f"https://{domain}/rest/api/3/filter/{PRODUCTION_FILTER_ID}",
            headers=headers, timeout=15,
        )
        if fr.status_code == 200:
            filter_data = fr.json()
            jql = filter_data.get("jql", "")
            filter_name = filter_data.get("name", "Production Issues")
        else:
            jql = "project = A20M AND type = Bug AND parent = A20M-2292 ORDER BY created DESC"
            filter_name = "Production Issues (fallback)"
    except requests.exceptions.RequestException:
        jql = "project = A20M AND type = Bug AND parent = A20M-2292 ORDER BY created DESC"
        filter_name = "Production Issues (fallback)"

    # Exclude closed issues — user wants only active ones
    # Splice status filter before any ORDER BY clause
    import re
    order_match = re.search(r'\bORDER\s+BY\b', jql, re.IGNORECASE)
    if order_match:
        head, tail = jql[:order_match.start()].rstrip(), jql[order_match.start():]
        jql = f'{head} AND statusCategory != Done {tail}'
    else:
        jql = f'{jql} AND statusCategory != Done'

    result = search_issues(jql, max_results=max_results)
    if isinstance(result, dict) and "error" in result:
        return result

    issues = result.get("issues", []) if isinstance(result, dict) else []

    # Aggregate stats
    by_status = {}
    by_severity = {}
    for i in issues:
        s = i.get("status", "-")
        by_status[s] = by_status.get(s, 0) + 1
        # severity often stored in priority for bugs
        p = i.get("priority", "-")
        by_severity[p] = by_severity.get(p, 0) + 1

    return {
        "issues": issues,
        "total": len(issues),
        "filter_name": filter_name,
        "jql": jql,
        "dashboard_url": PRODUCTION_DASHBOARD_URL,
        "by_status": sorted(by_status.items(), key=lambda x: -x[1]),
        "by_severity": sorted(by_severity.items(), key=lambda x: -x[1]),
    }


def universal_search(query, max_results=15):
    """
    Search JIRA globally: matches text in summary/description/comments + user names.
    Returns both matching issues and matching users.
    """
    query = (query or "").strip()
    if not query:
        return {"users": [], "issues": [], "jql": ""}

    users = search_users(query, max_results=5)

    # Escape quotes for JQL
    escaped = query.replace('\\', '\\\\').replace('"', '\\"')
    jql = f'text ~ "{escaped}" ORDER BY updated DESC'

    result = search_issues(jql, max_results=max_results)
    issues = result.get("issues", []) if isinstance(result, dict) and "error" not in result else []
    error = result.get("error") if isinstance(result, dict) else None

    return {
        "users": users,
        "issues": issues,
        "jql": jql,
        "query": query,
        "error_hint": error if not issues and not users else None,
    }


def add_comment(issue_key, body_text):
    """Post a comment to a JIRA issue"""
    headers = _auth_header()
    if not headers:
        return {"error": "JIRA credentials missing"}
    domain = os.environ.get('JIRA_DOMAIN', '')
    body_text = (body_text or "").strip()
    if not body_text:
        return {"error": "Comment body is empty"}
    payload = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [{"type": "text", "text": body_text}]
            }]
        }
    }
    try:
        r = requests.post(
            f"https://{domain}/rest/api/3/issue/{issue_key}/comment",
            headers=headers, json=payload, timeout=15,
        )
        if r.status_code in (200, 201):
            d = r.json()
            return {
                "ok": True,
                "id": d.get("id"),
                "author": (d.get("author") or {}).get("displayName", "You"),
                "created": (d.get("created") or "")[:10],
                "body": body_text,
            }
        return {"error": f"JIRA API {r.status_code}: {r.text[:250]}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


def add_worklog(issue_key, time_spent, comment=""):
    """
    Add worklog to a JIRA issue.
    time_spent format: "2h", "30m", "1h 30m", "1d"
    """
    headers = _auth_header()
    if not headers:
        return {"error": "JIRA credentials missing"}
    domain = os.environ.get('JIRA_DOMAIN', '')

    body = {"timeSpent": time_spent.strip()}
    if comment.strip():
        body["comment"] = {
            "type": "doc",
            "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [{"type": "text", "text": comment.strip()}]
            }]
        }

    try:
        r = requests.post(
            f"https://{domain}/rest/api/3/issue/{issue_key}/worklog",
            headers=headers,
            json=body,
            timeout=15,
        )
        if r.status_code in (200, 201):
            data = r.json()
            return {
                "ok": True,
                "id": data.get("id"),
                "timeSpent": data.get("timeSpent"),
                "timeSpentSeconds": data.get("timeSpentSeconds"),
                "created": (data.get("created") or "")[:10],
            }
        return {"error": f"JIRA API {r.status_code}: {r.text[:250]}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


def get_worklogs(issue_key, limit=5):
    """Fetch recent worklogs for an issue"""
    headers = _auth_header()
    if not headers:
        return []
    domain = os.environ.get('JIRA_DOMAIN', '')
    try:
        r = requests.get(
            f"https://{domain}/rest/api/3/issue/{issue_key}/worklog",
            headers=headers,
            timeout=15,
        )
        if r.status_code != 200:
            return []
        worklogs = r.json().get("worklogs", [])[-limit:]
        return [{
            "author": (w.get("author") or {}).get("displayName", "Unknown"),
            "created": (w.get("created") or "")[:10],
            "timeSpent": w.get("timeSpent", ""),
            "comment": _adf_to_text(w.get("comment")),
        } for w in worklogs]
    except requests.exceptions.RequestException:
        return []


def get_team_workload(members):
    """
    For given team members, return open task counts and recent tasks per member.
    One efficient JQL call for the whole team.
    """
    headers = _auth_header()
    if not headers or not members:
        return {"error": "No members or missing credentials"} if not headers else []
    domain = os.environ.get('JIRA_DOMAIN', '')

    account_ids = [m["accountId"] for m in members]
    ids_quoted = ", ".join(f'"{a}"' for a in account_ids)
    jql = (f'assignee in ({ids_quoted}) AND '
           f'statusCategory != Done '
           f'ORDER BY updated DESC')

    try:
        r = requests.get(
            f"https://{domain}/rest/api/3/search/jql",
            headers=headers,
            params={"jql": jql, "maxResults": 100,
                    "fields": "summary,status,assignee,priority,updated,duedate"},
            timeout=20,
        )
        if r.status_code != 200:
            return {"error": f"JIRA API {r.status_code}: {r.text[:200]}"}
        data = r.json()

        # Aggregate by assignee
        per_user = {m["accountId"]: {
            "member": m,
            "open_count": 0,
            "high_priority_count": 0,
            "overdue_count": 0,
            "latest_task": None,
        } for m in members}

        import datetime
        today = datetime.date.today().isoformat()

        for issue in data.get("issues", []):
            f = issue.get("fields", {})
            aid = ((f.get("assignee") or {}).get("accountId")) or ""
            if aid not in per_user:
                continue
            per_user[aid]["open_count"] += 1
            priority = (f.get("priority") or {}).get("name", "").lower()
            if priority in ("highest", "high"):
                per_user[aid]["high_priority_count"] += 1
            duedate = f.get("duedate")
            if duedate and duedate < today:
                per_user[aid]["overdue_count"] += 1
            if per_user[aid]["latest_task"] is None:
                per_user[aid]["latest_task"] = {
                    "key": issue.get("key"),
                    "summary": f.get("summary", ""),
                    "status": (f.get("status") or {}).get("name", "-"),
                    "updated": (f.get("updated") or "")[:10],
                    "url": f"https://{domain}/browse/{issue.get('key')}",
                }

        # Sort: idle first (highlight), then by open_count desc
        result = list(per_user.values())
        result.sort(key=lambda x: (x["open_count"] > 0, -x["open_count"]))
        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


def get_dashboard():
    """Aggregate analysis: my open tasks by status/priority/project"""
    headers = _auth_header()
    if not headers:
        return {"error": "JIRA credentials missing"}
    domain = os.environ.get('JIRA_DOMAIN', '')

    jql = "assignee = currentUser() ORDER BY updated DESC"
    try:
        r = requests.get(
            f"https://{domain}/rest/api/3/search/jql",
            headers=headers,
            params={"jql": jql, "maxResults": 100,
                    "fields": "summary,status,priority,issuetype,project,updated"},
            timeout=20,
        )
        if r.status_code != 200:
            return {"error": f"JIRA API {r.status_code}: {r.text[:200]}"}
        data = r.json()
        issues = data.get("issues", [])

        by_status = {}
        by_priority = {}
        by_project = {}
        by_type = {}
        recent = []
        open_count = 0
        done_count = 0

        for i in issues:
            f = i.get("fields", {})
            status = (f.get("status") or {}).get("name", "-")
            priority = (f.get("priority") or {}).get("name", "-")
            project = (f.get("project") or {}).get("key", "-")
            itype = (f.get("issuetype") or {}).get("name", "-")

            by_status[status] = by_status.get(status, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_project[project] = by_project.get(project, 0) + 1
            by_type[itype] = by_type.get(itype, 0) + 1

            if status.lower() in ("done", "closed", "resolved"):
                done_count += 1
            else:
                open_count += 1

            if len(recent) < 5:
                recent.append({
                    "key": i.get("key"),
                    "summary": f.get("summary", ""),
                    "status": status,
                    "priority": priority,
                    "updated": (f.get("updated") or "")[:10],
                    "url": f"https://{domain}/browse/{i.get('key')}",
                })

        return {
            "total": len(issues),
            "open": open_count,
            "done": done_count,
            "by_status": sorted(by_status.items(), key=lambda x: -x[1]),
            "by_priority": sorted(by_priority.items(), key=lambda x: -x[1]),
            "by_project": sorted(by_project.items(), key=lambda x: -x[1]),
            "by_type": sorted(by_type.items(), key=lambda x: -x[1]),
            "recent": recent,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


def parse_jira_command(cmd):
    """
    Parse chat command like:
      /jira MREC          → project = MREC, my tasks
      /jira MREC open     → open tasks in MREC
      /jira mine          → all tasks assigned to me
      /jira A20M-2797     → specific issue
      /jira jql project = MSPC AND status = Open
    """
    cmd = cmd.strip()
    if cmd.startswith("/jira"):
        cmd = cmd[5:].strip()

    if not cmd:
        return "assignee = currentUser() ORDER BY updated DESC"

    # JQL passthrough
    if cmd.lower().startswith("jql "):
        return cmd[4:].strip()

    # "mine"
    if cmd.lower() == "mine":
        return "assignee = currentUser() ORDER BY updated DESC"

    # "today" — updated today or due today
    if cmd.lower() in ("today", "today work", "today's work", "today task", "todays task"):
        return ("assignee = currentUser() AND "
                "(updated >= startOfDay() OR duedate = now()) "
                "ORDER BY updated DESC")

    # "week" — this week's work
    if cmd.lower() in ("week", "this week", "weekly"):
        return ("assignee = currentUser() AND "
                "updated >= startOfWeek() "
                "ORDER BY updated DESC")

    # Specific issue key like A20M-2797
    if "-" in cmd and cmd.split("-")[0].isalpha() and cmd.split("-")[1].isdigit():
        return f"key = {cmd.upper()}"

    # Project + optional status
    parts = cmd.split()
    project = parts[0].upper()
    if len(parts) > 1:
        status_word = parts[1].lower()
        status_map = {
            "open": "status = Open",
            "todo": 'status = "To Do"',
            "progress": 'status = "In Progress"',
            "done": "status = Done",
            "closed": "status = Closed",
        }
        status_clause = status_map.get(status_word, f'status = "{parts[1]}"')
        return f"project = {project} AND {status_clause} ORDER BY updated DESC"

    return f"project = {project} AND assignee = currentUser() ORDER BY updated DESC"
