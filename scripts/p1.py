import sys

f = r'C:\Users\Administrator\deepcoin\frontend\app\admin\page.tsx'
c = open(f, encoding='utf-8').read()

# Patch 1: update stats query
old1 = """    queryKey:  ["admin", "stats"],
    queryFn:   getAdminStats,
    enabled:   isPrivileged && authed,
    // WHY 60 s staleTime: route distribution changes slowly;
    // avoids a redundant refetch every time the tab is re-focused.
    staleTime: 60_000,
    // WHY retry 1: admin/stats returns 401 if called too early (before
    // SessionSync sets _authToken). One retry is enough to recover;
    // beyond that it\u2019s a real auth problem best surfaced to the user.
    retry:     1,
  });"""
new1 = """    queryKey:        ["admin", "stats"],
    queryFn:         getAdminStats,
    enabled:         isPrivileged && authed,
    // WHY 30 s refetchInterval: live activity feed + today KPIs refresh
    // regularly so admins see new analyses as they arrive.
    refetchInterval: 30_000,
    staleTime:       30_000,
    retry:           1,
  });"""

if old1 in c:
    c = c.replace(old1, new1, 1)
    print("ok1")
else:
    idx = c.find("WHY 60 s staleTime")
    print("FAIL1", repr(c[max(0,idx-50):idx+200]))
    sys.exit(1)

open(f, 'w', encoding='utf-8').write(c)
print("saved")
