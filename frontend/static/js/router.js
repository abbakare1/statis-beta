/**
 * STATIS Router
 * Hash-based routing for SPA navigation.
 * Supports deep links and browser back/forward.
 */

const STATIS_ROUTER = (() => {
    'use strict';

    const routes = new Map();
    let currentRoute = null;
    let notFoundHandler = null;

    /**
     * Register a route handler.
     * Pattern examples:
     *   'dashboard'           → #/dashboard
     *   'project/:id'         → #/project/abc123
     *   'project/:id/:sub'    → #/project/abc123/analysis
     */
    function register(pattern, handler) {
        const keys = [];
        const regex = new RegExp(
            '^' +
            pattern
                .split('/')
                .map(segment => {
                    if (segment.startsWith(':')) {
                        keys.push(segment.slice(1));
                        return '([^/]+)';
                    }
                    return segment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                })
                .join('/') +
            '$'
        );
        routes.set(pattern, { regex, keys, handler });
    }

    /**
     * Set the not-found handler.
     */
    function setNotFound(handler) {
        notFoundHandler = handler;
    }

    /**
     * Navigate to a route.
     */
    function navigate(path, replace = false) {
        const hash = '#' + (path.startsWith('/') ? path : '/' + path);
        if (replace) {
            window.history.replaceState(null, '', hash);
            handleRoute();
        } else {
            window.location.hash = hash;
        }
    }

    /**
     * Get current route parameters.
     */
    function getParams() {
        return currentRoute ? currentRoute.params : {};
    }

    /**
     * Get current route path.
     */
    function getPath() {
        return currentRoute ? currentRoute.path : '';
    }

    /**
     * Handle hash change and dispatch to matching route.
     */
    function handleRoute() {
        const hash = window.location.hash.slice(1) || '/dashboard';
        let path = hash.split('?')[0];
        if (path.startsWith('/')) {
            path = path.slice(1);
        }
        const query = Object.fromEntries(new URLSearchParams(hash.split('?')[1] || ''));

        for (const [pattern, route] of routes) {
            const match = path.match(route.regex);
            if (match) {
                const params = {};
                route.keys.forEach((key, i) => {
                    params[key] = decodeURIComponent(match[i + 1]);
                });
                currentRoute = { path, pattern, params, query };
                try {
                    route.handler(params, query);
                } catch (e) {
                    console.error('Route handler error:', e);
                }
                return;
            }
        }

        currentRoute = { path, pattern: null, params: {}, query };
        if (notFoundHandler) {
            notFoundHandler(path);
        } else {
            navigate('/dashboard', true);
        }
    }

    /**
     * Initialize the router.
     */
    function init() {
        window.addEventListener('hashchange', handleRoute);
        handleRoute();
    }

    return {
        register,
        setNotFound,
        navigate,
        getParams,
        getPath,
        init,
    };
})();

if (typeof window !== 'undefined') window.STATIS_ROUTER = STATIS_ROUTER;
