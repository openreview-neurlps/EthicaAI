#!/usr/bin/env python3
"""EthicaAI site/ 폴더의 모든 HTML에 PostHog 추적 스니펫을 자동 주입.

빌드 후 또는 커밋 전에 실행하면 됩니다.
이미 주입된 파일은 자동으로 skip합니다.

사용법:
    python scripts/inject_posthog.py
"""

import glob
import os

POSTHOG_SNIPPET = """
    <!-- PostHog Analytics (auto-injected) -->
    <script>
        !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags getFeatureFlag getFeatureFlagPayload reloadFeatureFlags group updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures getActiveMatchingSurveys getSurveys onSessionId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);
        posthog.init('phc_158CbBeWD8X1eNyD4xpi8VklVxsNZtx5yclocxfpgiO', {api_host:'https://us.i.posthog.com', person_profiles: 'identified_only'});
    </script>
"""

MARKER = "phc_158CbBeWD8X1eNyD4xpi8VklVxsNZtx5yclocxfpgiO"


def inject(site_dir: str) -> None:
    """site 디렉토리 내 모든 HTML에 PostHog 주입."""
    html_files = glob.glob(os.path.join(site_dir, "**/*.html"), recursive=True)
    injected, skipped = 0, 0

    for path in html_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            skipped += 1
            continue

        if MARKER in content:
            skipped += 1
            continue

        if "</head>" in content:
            content = content.replace("</head>", POSTHOG_SNIPPET + "\n</head>")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            injected += 1

    print(f"✅ PostHog injected: {injected} files | Skipped (already has): {skipped}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    site_dir = os.path.join(os.path.dirname(script_dir), "site")
    inject(site_dir)
