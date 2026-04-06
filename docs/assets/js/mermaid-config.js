document.addEventListener('DOMContentLoaded', function() {
  if (typeof mermaid !== 'undefined') {
    const isDark = document.body.getAttribute('data-md-color-scheme') === 'slate';

    mermaid.initialize({
      startOnLoad: true,
      theme: isDark ? 'base' : 'default',
      securityLevel: 'loose',
      themeVariables: isDark ? {
        background: '#0d1117',
        mainBkg: '#1e293b',
        primaryColor: '#3b82f6',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#60a5fa',
        secondaryColor: '#10b981',
        secondaryTextColor: '#ffffff',
        secondaryBorderColor: '#34d399',
        tertiaryColor: '#8b5cf6',
        tertiaryTextColor: '#ffffff',
        lineColor: '#22d3ee',
        textColor: '#ffffff',
        nodeTextColor: '#ffffff',
        clusterBkg: '#0f172a',
        clusterBorder: '#475569',
        edgeLabelBackground: '#334155',
        fontFamily: 'Inter, -apple-system, sans-serif',
        fontSize: '14px',
        actorBorder: '#60a5fa',
        actorBkg: '#1e293b',
        actorTextColor: '#ffffff',
        actorLineColor: '#22d3ee',
        signalColor: '#22d3ee',
        signalTextColor: '#ffffff',
        noteBorderColor: '#fbbf24',
        noteBkgColor: '#1e293b',
        noteTextColor: '#ffffff'
      } : {},
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis',
        padding: 20,
        nodeSpacing: 50,
        rankSpacing: 60
      },
      sequence: {
        useMaxWidth: true,
        wrap: true,
        width: 200,
        height: 60,
        mirrorActors: true,
        actorFontSize: 14,
        actorFontWeight: 600,
        noteFontSize: 13,
        messageFontSize: 14,
        messageFontWeight: 500
      }
    });
  }
});
