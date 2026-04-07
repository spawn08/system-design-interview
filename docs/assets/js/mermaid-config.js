function initMermaid() {
  if (typeof mermaid === 'undefined') return;

  const isDark = document.body.getAttribute('data-md-color-scheme') === 'slate';

  mermaid.initialize({
    startOnLoad: false,
    theme: isDark ? 'base' : 'default',
    securityLevel: 'loose',
    themeVariables: isDark ? {
      background: '#0c0a1d',
      mainBkg: 'rgba(124, 58, 237, 0.12)',
      primaryColor: '#7c3aed',
      primaryTextColor: '#ffffff',
      primaryBorderColor: '#a78bfa',
      secondaryColor: '#06b6d4',
      secondaryTextColor: '#ffffff',
      secondaryBorderColor: '#22d3ee',
      tertiaryColor: '#ec4899',
      tertiaryTextColor: '#ffffff',
      lineColor: '#22d3ee',
      textColor: '#ffffff',
      nodeTextColor: '#ffffff',
      clusterBkg: 'rgba(12, 10, 29, 0.5)',
      clusterBorder: 'rgba(124, 58, 237, 0.15)',
      edgeLabelBackground: 'rgba(42, 37, 69, 0.9)',
      fontFamily: 'Inter, -apple-system, sans-serif',
      fontSize: '14px',
      actorBorder: '#a78bfa',
      actorBkg: 'rgba(124, 58, 237, 0.12)',
      actorTextColor: '#ffffff',
      actorLineColor: '#22d3ee',
      signalColor: '#22d3ee',
      signalTextColor: '#ffffff',
      noteBorderColor: '#f59e0b',
      noteBkgColor: 'rgba(245, 158, 11, 0.1)',
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

  renderMermaidBlocks();
}

function renderMermaidBlocks() {
  if (typeof mermaid === 'undefined') return;

  document.querySelectorAll('div.mermaid').forEach(function (el) {
    if (el.getAttribute('data-mermaid-processed')) return;

    var raw = el.textContent.trim();
    if (!raw) return;

    el.setAttribute('data-mermaid-processed', 'true');
    el.removeAttribute('data-processed');
    el.innerHTML = raw;
  });

  try {
    mermaid.run({ querySelector: 'div.mermaid:not([data-processed])' });
  } catch (_) {
    // Fallback for older mermaid.run signatures
    mermaid.init(undefined, 'div.mermaid:not([data-processed])');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMermaid);
} else {
  initMermaid();
}

// Material for MkDocs instant navigation re-renders
if (typeof document$ !== 'undefined') {
  document$.subscribe(function () {
    initMermaid();
  });
}
