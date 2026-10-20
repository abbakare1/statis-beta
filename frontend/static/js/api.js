/**
 * STATIS API Service Layer
 * Centralizes all HTTP requests, error handling, and response parsing.
 * Phase 2I: Frontend must never perform statistical computation.
 */

const STATIS_API = (() => {
    'use strict';

    const BASE = '';
    let _requestCount = 0;
    const _inFlight = new Map();

    /**
     * Core request function with error handling and deduplication.
     */
    async function request(method, path, options = {}) {
        const url = BASE + path;
        const key = method + ':' + url + ':' + JSON.stringify(options.body || null);

        // Request deduplication for GET requests
        if (method === 'GET' && _inFlight.has(key)) {
            return _inFlight.get(key);
        }

        const fetchOptions = {
            method,
            headers: { ...(options.headers || {}) },
            ...options,
        };

        if (options.body && !(options.body instanceof FormData)) {
            if (typeof options.body === 'object') {
                fetchOptions.headers['Content-Type'] = 'application/json';
                fetchOptions.body = JSON.stringify(options.body);
            } else {
                fetchOptions.body = options.body;
            }
        }

        _requestCount++;
        const promise = (async () => {
            try {
                const response = await fetch(url, fetchOptions);
                const contentType = response.headers.get('content-type') || '';
                let data = null;

                if (contentType.includes('application/json')) {
                    data = await response.json();
                } else if (response.ok) {
                    data = await response.blob();
                } else {
                    const text = await response.text();
                    data = { detail: text || response.statusText };
                }

                if (!response.ok) {
                    throw new APIError(
                        data?.detail || data?.message || response.statusText,
                        response.status,
                        data?.error_code || data?.detail?.error_code,
                        data
                    );
                }

                return data;
            } catch (error) {
                if (error instanceof APIError) throw error;
                // Network error
                throw new APIError(
                    'Unable to connect to STATIS. Please check that the server is running.',
                    0,
                    'NETWORK_ERROR',
                    { original: error.message }
                );
            } finally {
                if (method === 'GET') _inFlight.delete(key);
            }
        })();

        if (method === 'GET') {
            _inFlight.set(key, promise);
        }

        return promise;
    }

    /**
     * Custom API error class with researcher-friendly messages.
     */
    class APIError extends Error {
        constructor(message, status, errorCode, payload) {
            super(message);
            this.name = 'APIError';
            this.status = status;
            this.errorCode = errorCode;
            this.payload = payload;
        }

        get isNetworkError() { return this.status === 0; }
        get isNotFound() { return this.status === 404; }
        get isConflict() { return this.status === 409; }
        get isValidationError() { return this.status === 422; }
        get isLifecycleError() { return this.status === 409 && ['INVALID_OUTPUT_STATE', 'APPROVAL_NOT_ALLOWED', 'ARCHIVE_NOT_ALLOWED'].includes(this.errorCode); }
    }

    /**
     * Translate backend error to researcher-facing message.
     */
    function humanizeError(error) {
        if (!(error instanceof APIError)) return 'An unexpected error occurred.';
        if (error.isNetworkError) return error.message;

        const code = error.errorCode;
        const messages = {
            'OUTPUT_NOT_FOUND': 'The requested output could not be found.',
            'INVALID_OUTPUT_STATE': 'This action is not allowed in the current output state.',
            'VALIDATION_FAILED': 'Validation failed. Please review the output specification.',
            'DATASET_NOT_FOUND': 'The dataset could not be found.',
            'DATASET_IDENTITY_MISMATCH': 'Dataset identity mismatch. The dataset has changed since the analysis was created.',
            'ANALYSIS_VERSION_NOT_FOUND': 'The analysis version could not be found.',
            'ARTIFACT_NOT_FOUND': 'The artifact could not be found.',
            'ARTIFACT_INTEGRITY_FAILURE': 'Artifact integrity check failed.',
            'APPROVAL_NOT_ALLOWED': 'This output cannot be approved in its current state.',
            'ARCHIVE_NOT_ALLOWED': 'This output cannot be archived in its current state.',
            'CROSS_PROJECT_BINDING': 'Cross-project binding is not allowed.',
            'PATH_TRAVERSAL_BLOCKED': 'Invalid artifact path.',
            'RENDERING_FAILURE': 'Rendering failed. Please check the output specification.',
            'NOT_FOUND': 'The requested resource was not found.',
            'BLOCK': 'This analysis has validation blockers that must be resolved before approval.',
        };

        return messages[code] || error.message || 'An error occurred.';
    }

    // -----------------------------------------------------------------------
    // Projects
    // -----------------------------------------------------------------------
    const projects = {
        list: () => request('GET', '/api/projects'),
        summary: () => request('GET', '/api/projects/summary'),
        get: (id) => request('GET', `/api/projects/${id}`),
        create: (data) => {
            const fd = new FormData();
            fd.append('name', data.name);
            fd.append('description', data.description || '');
            fd.append('research_question', data.research_question || '');
            fd.append('study_design', data.study_design || '');
            return request('POST', '/api/projects', { body: fd });
        },
        workflow: (id) => request('GET', `/api/projects/${id}/workflow`),
        plans: (id) => request('GET', `/api/projects/${id}/plans`),
        runs: (id) => request('GET', `/api/projects/${id}/runs`),
        outputs: (id) => request('GET', `/api/projects/${id}/outputs`),
        variableRoles: (id) => request('GET', `/api/projects/${id}/variable-roles`),
        qualityProfile: (id) => request('GET', `/api/projects/${id}/quality-profile`),
        reproducibility: (id) => request('GET', `/api/projects/${id}/reproducibility`),
    };

    // -----------------------------------------------------------------------
    // Datasets
    // -----------------------------------------------------------------------
    const datasets = {
        list: (projectId) => request('GET', `/api/projects/${projectId}/datasets`),
        upload: (projectId, file) => {
            const fd = new FormData();
            fd.append('file', file);
            return request('POST', `/api/projects/${projectId}/upload`, { body: fd });
        },
        preview: (datasetId, head = 10) => request('GET', `/api/datasets/${datasetId}/preview?head=${head}`),
    };

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------
    const pipeline = {
        run: (body) => request('POST', '/api/pipeline/run', { body }),
    };

    // -----------------------------------------------------------------------
    // Data Quality
    // -----------------------------------------------------------------------
    const quality = {
        assess: (datasetId, body) => request('POST', `/api/data-quality/${datasetId}/assess`, { body }),
        get: (datasetId) => request('GET', `/api/data-quality/${datasetId}`),
        missingness: (datasetId) => request('GET', `/api/data-quality/${datasetId}/missingness`),
        outliers: (datasetId) => request('GET', `/api/data-quality/${datasetId}/outliers`),
    };

    // -----------------------------------------------------------------------
    // Analysis Plans
    // -----------------------------------------------------------------------
    const analysisPlan = {
        create: (body) => request('POST', '/api/analysis-plan/create', { body }),
        get: (planId) => request('GET', `/api/analysis-plan/${planId}`),
        approve: (analysisId, planId) => request('POST', `/api/analysis-plan/${analysisId}/approve`, { body: { plan_id: planId } }),
        reject: (analysisId, planId, reason) => request('POST', `/api/analysis-plan/${analysisId}/reject`, { body: { plan_id: planId, reason } }),
        lock: (analysisId, planId) => request('POST', `/api/analysis-plan/${analysisId}/lock`, { body: { plan_id: planId } }),
        execute: (analysisId, planId) => request('POST', `/api/analysis-plan/${analysisId}/execute`, { body: { plan_id: planId } }),
        decisionTrace: (analysisId) => request('GET', `/api/analysis-plan/${analysisId}/decision-trace`),
    };

    // -----------------------------------------------------------------------
    // Analysis Runs
    // -----------------------------------------------------------------------
    const runs = {
        list: (params = {}) => {
            const qs = new URLSearchParams(params).toString();
            return request('GET', `/api/runs${qs ? '?' + qs : ''}`);
        },
        get: (runId) => request('GET', `/api/runs/${runId}`),
        verify: (body) => request('POST', '/api/runs/verify', { body }),
        compare: (body) => request('POST', '/api/runs/compare', { body }),
    };

    // -----------------------------------------------------------------------
    // Outputs (Phase 2G/2H)
    // -----------------------------------------------------------------------
    function buildOutputApi(type) {
        const base = `/api/outputs/${type}s`;
        return {
            list: (params = {}) => {
                const qs = new URLSearchParams(
                    Object.entries(params).filter(([, v]) => v != null && v !== '')
                ).toString();
                return request('GET', `${base}${qs ? '?' + qs : ''}`);
            },
            get: (id) => request('GET', `${base}/${id}`),
            create: (body) => request('POST', base, { body }),
            validate: (id) => request('POST', `${base}/${id}/validate`),
            render: (id) => request('POST', `${base}/${id}/render`),
            approve: (id, body) => request('POST', `${base}/${id}/approve`, { body: body || {} }),
            review: (id) => request('POST', `${base}/${id}/review`),
            archive: (id) => request('POST', `${base}/${id}/archive`),
            artifact: (id) => request('GET', `${base}/${id}/artifact`),
            downloadUrl: (id) => `${BASE}${base}/${id}/artifact/download`,
        };
    }

    const figures = buildOutputApi('figure');
    const tables = buildOutputApi('table');
    const reports = buildOutputApi('report');

    const reproducibility = {
        create: (body) => request('POST', '/api/outputs/reproducibility', { body }),
        get: (packageId) => request('GET', `/api/outputs/reproducibility/${packageId}`),
        manifest: (packageId) => request('GET', `/api/outputs/reproducibility/${packageId}/manifest`),
        verify: (packageId, body) => request('POST', `/api/outputs/reproducibility/${packageId}/verify`, { body: body || {} }),
        artifact: (packageId) => request('GET', `/api/outputs/reproducibility/${packageId}/artifact`),
    };

    // -----------------------------------------------------------------------
    // Audit
    // -----------------------------------------------------------------------
    const audit = {
        list: (params = {}) => {
            const qs = new URLSearchParams(
                Object.entries(params).filter(([, v]) => v != null && v !== '')
            ).toString();
            return request('GET', `/api/audit${qs ? '?' + qs : ''}`);
        },
    };

    // -----------------------------------------------------------------------
    // Methods
    // -----------------------------------------------------------------------
    const methods = {
        list: () => request('GET', '/api/methods'),
    };

    // -----------------------------------------------------------------------
    // Health
    // -----------------------------------------------------------------------
    const health = {
        check: () => request('GET', '/api/health'),
    };

    return {
        APIError,
        humanizeError,
        get requestCount() { return _requestCount; },
        projects,
        datasets,
        pipeline,
        quality,
        analysisPlan,
        runs,
        figures,
        tables,
        reports,
        reproducibility,
        audit,
        methods,
        health,
    };
})();

// Export for module systems / global access
if (typeof window !== 'undefined') window.STATIS_API = STATIS_API;
