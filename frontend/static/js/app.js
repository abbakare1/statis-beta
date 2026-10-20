/**
 * STATIS Frontend — Main Entry Point
 * Phase 2I: Researcher-facing statistical analysis workspace.
 * All statistical computation remains in the deterministic backend.
 */

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    const STATE = window.STATIS_STATE;
    const ROUTER = window.STATIS_ROUTER;
    const UI = window.STATIS_UI;
    const API = window.STATIS_API;
    const VIEWS = window.STATIS_VIEWS;

    // =====================================================================
    // Notification rendering
    // =====================================================================
    STATE.on('notifications', (notifications) => {
        const container = document.getElementById('notifications');
        if (!container) return;
        container.innerHTML = notifications.map(n => UI.notificationHtml(n)).join('');
    });

    // =====================================================================
    // Loading state indicators
    // =====================================================================
    STATE.on('loading', (loadingSet) => {
        document.querySelectorAll('[data-loading-key]').forEach(el => {
            const key = el.dataset.loadingKey;
            el.classList.toggle('is-loading', loadingSet.has(key));
        });
    });

    // =====================================================================
    // Router setup
    // =====================================================================
    ROUTER.register('dashboard', () => VIEWS.dashboard());
    ROUTER.register('project/:id', (params) => {
        STATE.set('currentSubView', 'overview');
        VIEWS.project(params.id);
    });
    ROUTER.register('project/:id/:sub', (params) => {
        STATE.set('currentSubView', params.sub || 'overview');
        VIEWS.project(params.id);
    });
    ROUTER.register('results', () => VIEWS.resultsGlobal());
    ROUTER.register('outputs', () => VIEWS.outputsGlobal());
    ROUTER.register('reports', () => VIEWS.reportsGlobal());
    ROUTER.register('audit', () => VIEWS.auditGlobal());

    ROUTER.setNotFound(() => ROUTER.navigate('/dashboard', true));

    // =====================================================================
    // Global navigation
    // =====================================================================
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const view = link.dataset.view;
            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => {
                l.classList.toggle('active', l === link);
                l.setAttribute('aria-current', l === link ? 'page' : 'false');
            });
            // Navigate
            switch (view) {
                case 'dashboard': ROUTER.navigate('/dashboard'); break;
                case 'results': ROUTER.navigate('/results'); break;
                case 'outputs': ROUTER.navigate('/outputs'); break;
                case 'reports': ROUTER.navigate('/reports'); break;
                case 'audit': ROUTER.navigate('/audit'); break;
            }
        });
    });

    // =====================================================================
    // Keyboard shortcuts
    // =====================================================================
    document.addEventListener('keydown', (e) => {
        // Escape closes modals
        if (e.key === 'Escape') {
            const modal = document.getElementById('modal-overlay');
            if (modal && !modal.classList.contains('hidden')) {
                VIEWS.closeModal();
            }
        }
    });

    // =====================================================================
    // Initialize
    // =====================================================================
    ROUTER.init();

    // Health check (non-blocking)
    API.health.check().then(h => {
        if (h && h.provider_available === false) {
            STATE.notify('LLM model provider is not available. Some AI-assisted features will be limited.', 'warning', 10000);
        }
    }).catch(() => {
        // Silently ignore — health check is informational
    });
});
