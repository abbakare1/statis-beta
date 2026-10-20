/**
 * STATIS State Management
 * Centralized application state with subscription-based updates.
 * Phase 2I: State mirrors backend; frontend never fabricates state.
 */

const STATIS_STATE = (() => {
    'use strict';

    const state = {
        // Navigation
        currentView: 'dashboard',
        currentProjectId: null,
        currentSubView: 'overview',

        // Data
        projects: [],
        projectSummary: null,
        projectDetail: null,
        projectWorkflow: null,
        datasets: [],
        plans: [],
        runs: [],
        outputs: null,
        variableRoles: null,
        qualityProfiles: null,
        reproPackages: null,
        auditEntries: [],
        methods: [],

        // UI state
        loading: new Set(),
        errors: {},
        notifications: [],
    };

    const listeners = new Map();

    /**
     * Subscribe to state changes for a specific key.
     */
    function on(key, callback) {
        if (!listeners.has(key)) listeners.set(key, new Set());
        listeners.get(key).add(callback);
        return () => listeners.get(key).delete(callback);
    }

    /**
     * Update state and notify listeners.
     */
    function set(key, value) {
        const old = state[key];
        state[key] = value;
        if (listeners.has(key)) {
            listeners.get(key).forEach(cb => {
                try { cb(value, old, key); } catch (e) { console.error('State listener error:', e); }
            });
        }
    }

    /**
     * Merge object into state key.
     */
    function merge(key, obj) {
        const current = state[key];
        if (current && typeof current === 'object' && !Array.isArray(current)) {
            set(key, { ...current, ...obj });
        } else {
            set(key, obj);
        }
    }

    /**
     * Mark a key as loading.
     */
    function startLoading(key) {
        const s = new Set(state.loading);
        s.add(key);
        set('loading', s);
    }

    /**
     * Mark a key as not loading.
     */
    function endLoading(key) {
        const s = new Set(state.loading);
        s.delete(key);
        set('loading', s);
    }

    function isLoading(key) {
        return state.loading.has(key);
    }

    /**
     * Set an error for a key.
     */
    function setError(key, error) {
        const errors = { ...state.errors };
        if (error) {
            errors[key] = error;
        } else {
            delete errors[key];
        }
        set('errors', errors);
    }

    function getError(key) {
        return state.errors[key] || null;
    }

    /**
     * Add a notification.
     */
    function notify(message, type = 'info', ttl = 5000) {
        const id = Date.now() + Math.random();
        const notification = { id, message, type };
        set('notifications', [...state.notifications, notification]);
        if (ttl > 0) {
            setTimeout(() => {
                set('notifications', state.notifications.filter(n => n.id !== id));
            }, ttl);
        }
        return id;
    }

    function dismissNotification(id) {
        set('notifications', state.notifications.filter(n => n.id !== id));
    }

    /**
     * Clear project-scoped state.
     */
    function clearProject() {
        set('projectDetail', null);
        set('projectWorkflow', null);
        set('datasets', []);
        set('plans', []);
        set('runs', []);
        set('outputs', null);
        set('variableRoles', null);
        set('qualityProfiles', null);
        set('reproPackages', null);
    }

    /**
     * Get a derived status label from backend state.
     * Never fabricates status — returns null if unknown.
     */
    function statusLabel(status) {
        const map = {
            'created': 'Not Started',
            'in_progress': 'In Progress',
            'complete': 'Complete',
            'needs_review': 'Needs Review',
            'not_started': 'Not Started',
            'draft': 'Draft',
            'validating': 'Validating',
            'validated': 'Validated',
            'rendering': 'Rendering',
            'generated': 'Generated',
            'review_required': 'Needs Review',
            'approved': 'Approved',
            'archived': 'Archived',
            'validation_failed': 'Validation Failed',
            'rendering_failed': 'Rendering Failed',
            'proposed': 'Proposed',
            'locked': 'Locked',
            'executed': 'Executed',
            'rejected': 'Rejected',
            'blocked': 'Blocked',
            'warning': 'Warning',
            'not_assessed': 'Not Assessed',
        };
        return map[status] || status || 'Unknown';
    }

    /**
     * Get status CSS class from backend state.
     */
    function statusClass(status) {
        const map = {
            'complete': 'complete',
            'approved': 'complete',
            'executed': 'complete',
            'validated': 'complete',
            'generated': 'complete',
            'archived': 'archived',
            'in_progress': 'progress',
            'rendering': 'progress',
            'validating': 'progress',
            'needs_review': 'review',
            'review_required': 'review',
            'proposed': 'review',
            'draft': 'draft',
            'not_started': 'draft',
            'created': 'draft',
            'warning': 'warning',
            'validation_failed': 'error',
            'rendering_failed': 'error',
            'rejected': 'error',
            'blocked': 'error',
            'not_assessed': 'unknown',
        };
        return map[status] || 'unknown';
    }

    return {
        state,
        on,
        set,
        merge,
        startLoading,
        endLoading,
        isLoading,
        setError,
        getError,
        notify,
        dismissNotification,
        clearProject,
        statusLabel,
        statusClass,
    };
})();

if (typeof window !== 'undefined') window.STATIS_STATE = STATIS_STATE;
