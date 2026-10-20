/**
 * STATIS UI Components
 * Reusable, accessible UI components for the researcher-facing interface.
 * All status indicators are textual — never color-only.
 */

const STATIS_UI = (() => {
    'use strict';

    const esc = (s) => {
        if (s == null) return '';
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    };

    /**
     * Loading spinner/message.
     */
    function loading(message = 'Loading...') {
        return `<div class="state-block state-loading" role="status" aria-live="polite">
            <div class="spinner" aria-hidden="true"></div>
            <p>${esc(message)}</p>
        </div>`;
    }

    /**
     * Empty state message with optional action.
     */
    function empty(message, actionHtml = '') {
        return `<div class="state-block state-empty">
            <p>${esc(message)}</p>
            ${actionHtml}
        </div>`;
    }

    /**
     * Error state with optional retry.
     */
    function error(message, retryAction = null) {
        return `<div class="state-block state-error" role="alert">
            <p>${esc(message)}</p>
            ${retryAction ? `<button class="btn btn-secondary btn-sm" onclick="${esc(retryAction)}">Retry</button>` : ''}
        </div>`;
    }

    /**
     * Status badge — textual, accessible.
     */
    function statusBadge(status, label = null) {
        const text = label || STATIS_STATE.statusLabel(status);
        const cls = STATIS_STATE.statusClass(status);
        return `<span class="badge badge-${cls}" role="status" aria-label="Status: ${esc(text)}">${esc(text)}</span>`;
    }

    /**
     * Workflow step indicator.
     */
    function workflowStep(step) {
        const state = step.state || 'not_started';
        const label = STATIS_STATE.statusLabel(state);
        const cls = STATIS_STATE.statusClass(state);
        const icon = state === 'complete' ? '✓' : state === 'needs_review' ? '⚠' : state === 'error' ? '✗' : '○';
        return `<div class="workflow-step workflow-${cls}" role="listitem" aria-label="${esc(step.label)}: ${esc(label)}">
            <span class="workflow-icon" aria-hidden="true">${icon}</span>
            <span class="workflow-label">${esc(step.label)}</span>
            <span class="workflow-status">${esc(label)}</span>
        </div>`;
    }

    /**
     * Expandable section (for technical details).
     */
    function expandable(title, contentHtml, open = false) {
        return `<details class="expandable" ${open ? 'open' : ''}>
            <summary class="expandable-summary">${esc(title)}</summary>
            <div class="expandable-content">${contentHtml}</div>
        </details>`;
    }

    /**
     * Key-value display row.
     */
    function kvRow(key, value, mono = false) {
        if (value == null || value === '') return '';
        return `<div class="kv-row">
            <span class="kv-key">${esc(key)}</span>
            <span class="kv-value ${mono ? 'mono' : ''}">${esc(String(value))}</span>
        </div>`;
    }

    /**
     * Metadata card header.
     */
    function cardHeader(title, badgeHtml = '', subtitle = '') {
        return `<div class="card-header">
            <div class="card-header-main">
                <h4 class="card-title">${esc(title)}</h4>
                ${subtitle ? `<p class="card-subtitle">${esc(subtitle)}</p>` : ''}
            </div>
            ${badgeHtml ? `<div class="card-header-badge">${badgeHtml}</div>` : ''}
        </div>`;
    }

    /**
     * Action button group.
     */
    function actionButtons(actions) {
        if (!actions || !actions.length) return '';
        return `<div class="action-group">` +
            actions.map(a => {
                if (a.hidden) return '';
                const cls = a.primary ? 'btn-primary' : a.danger ? 'btn-danger' : 'btn-secondary';
                const disabled = a.disabled ? 'disabled' : '';
                const onclick = a.onclick ? `onclick="${esc(a.onclick)}"` : '';
                return `<button class="btn ${cls} btn-sm" ${disabled} ${onclick} ${a.ariaLabel ? `aria-label="${esc(a.ariaLabel)}"` : ''}>${esc(a.label)}</button>`;
            }).join('') +
            `</div>`;
    }

    /**
     * Project card for dashboard.
     */
    function projectCard(project) {
        const status = project.workflow_status || project.status || 'unknown';
        const statusText = STATIS_STATE.statusLabel(status);
        const statusCls = STATIS_STATE.statusClass(status);
        const design = project.study_design || 'Study design not set';
        const datasets = project.dataset_count ?? 0;
        const analyses = project.analysis_run_count ?? 0;
        const plans = project.analysis_plan_count ?? 0;
        const outputs = project.output_count ?? 0;
        const reviewItems = project.review_items || [];

        const meta = [];
        if (datasets > 0) meta.push(`${datasets} dataset${datasets !== 1 ? 's' : ''}`);
        if (plans > 0) meta.push(`${plans} plan${plans !== 1 ? 's' : ''}`);
        if (analyses > 0) meta.push(`${analyses} analysis${analyses !== 1 ? 'es' : ''}`);
        if (outputs > 0) meta.push(`${outputs} output${outputs !== 1 ? 's' : ''}`);

        const reviewHtml = reviewItems.length > 0
            ? `<div class="card-review-items">
                <span class="review-label">Requires attention:</span>
                ${reviewItems.slice(0, 2).map(r => `<span class="review-item">${esc(r)}</span>`).join('')}
                ${reviewItems.length > 2 ? `<span class="review-item">+${reviewItems.length - 2} more</span>` : ''}
            </div>`
            : '';

        return `<article class="project-card" role="article">
            <div class="project-card-header">
                <h3 class="project-card-name">
                    <a href="#/project/${esc(project.id)}" class="project-link">${esc(project.name)}</a>
                </h3>
                <span class="badge badge-${statusCls}" role="status">${esc(statusText)}</span>
            </div>
            ${project.description ? `<p class="project-card-desc">${esc(project.description)}</p>` : ''}
            <div class="project-card-meta">
                <span class="meta-design">${esc(design)}</span>
                ${meta.length ? `<span class="meta-sep">·</span><span class="meta-counts">${meta.map(esc).join(' · ')}</span>` : ''}
            </div>
            ${reviewHtml}
            <div class="project-card-footer">
                <a href="#/project/${esc(project.id)}" class="btn btn-primary btn-sm">Open Project</a>
            </div>
        </article>`;
    }

    /**
     * Analysis card for plan view.
     */
    function analysisCard(analysis, planId) {
        const lifecycle = analysis.lifecycle || analysis.approval || 'draft';
        const validation = analysis.validation || 'pending';
        const approval = analysis.approval || 'proposed';
        const method = analysis.method || analysis.analysis_name || 'Unknown method';
        const name = analysis.analysis_name || method;
        const category = analysis.analysis_class || 'exploratory';
        const outcome = analysis.outcome || 'Not specified';
        const predictors = analysis.predictors || [];

        const concerns = analysis.warnings || [];
        const blockers = analysis.validation_blockers || [];

        const statusBadgeHtml = statusBadge(lifecycle);

        const concernsHtml = blockers.length > 0
            ? `<div class="analysis-concerns">
                <span class="concern-label">Blockers:</span>
                ${blockers.map(b => `<span class="concern-item concern-blocker">${esc(b)}</span>`).join('')}
            </div>`
            : concerns.length > 0
            ? `<div class="analysis-concerns">
                <span class="concern-label">Potential concerns:</span>
                ${concerns.map(c => `<span class="concern-item">${esc(c)}</span>`).join('')}
            </div>`
            : '';

        const canApprove = validation !== 'block' && approval !== 'approved' && lifecycle !== 'executed' && lifecycle !== 'locked';
        const canExecute = lifecycle === 'locked';

        const actions = [
            { label: 'Review', onclick: `STATIS_VIEWS.reviewAnalysis('${esc(planId)}', '${esc(analysis.analysis_id || '')}')`, primary: true },
            { label: 'Approve', onclick: `STATIS_VIEWS.approveAnalysis('${esc(planId)}', '${esc(analysis.analysis_id || '')}')`, disabled: !canApprove, hidden: lifecycle === 'executed' },
            { label: 'Execute', onclick: `STATIS_VIEWS.executeAnalysis('${esc(planId)}', '${esc(analysis.analysis_id || '')}')`, disabled: !canExecute, hidden: lifecycle !== 'locked' },
        ].filter(Boolean);

        return `<article class="analysis-card" role="article">
            ${cardHeader(name, statusBadgeHtml, method)}
            <div class="analysis-card-body">
                <div class="analysis-meta">
                    ${kvRow('Category', category)}
                    ${kvRow('Outcome', outcome)}
                    ${predictors.length ? kvRow('Predictors', predictors.join(', ')) : ''}
                    ${kvRow('Validation', validation === 'passed' ? 'Passed' : validation === 'block' ? 'Blocked' : validation === 'warning' ? 'Warning' : 'Pending')}
                    ${kvRow('Approval', approval === 'approved' ? 'Approved' : approval === 'rejected' ? 'Rejected' : 'Pending')}
                </div>
                ${concernsHtml}
            </div>
            <div class="analysis-card-footer">
                ${actionButtons(actions)}
            </div>
        </article>`;
    }

    /**
     * Statistical result display.
     * IMPORTANT: Displays values exactly as returned by backend.
     * Never calculates or modifies numerical values.
     */
    function resultDisplay(result) {
        if (!result) return empty('No result data available.');

        const method = result.method_name || result.method_id || 'Unknown method';
        const n = result.n_observed;
        const pValue = result.p_value;
        const effect = result.effect_size;
        const effectName = result.effect_size_name || 'Effect size';
        const ci = result.confidence_interval;
        const ciLevel = result.confidence_level || 0.95;
        const testStat = result.test_statistic;
        const df = result.degrees_of_freedom;
        const defensibility = result.defensibility_status || 'not_assessed';
        const execStatus = result.execution_status || 'unknown';
        const fingerprint = result.result_fingerprint;

        const defensibilityLabels = {
            'defensible': 'DEFENSIBLE',
            'defensible_with_caveats': 'DEFENSIBLE WITH CAVEATS',
            'review_required': 'REVIEW REQUIRED',
            'not_defensible': 'NOT DEFENSIBLE',
            'not_assessed': 'NOT ASSESSED',
        };

        const rows = [];
        if (n != null) rows.push({ label: 'N Observed', value: n.toLocaleString() });
        if (effect != null) rows.push({ label: effectName, value: String(effect) });
        if (ci && Array.isArray(ci)) rows.push({ label: `${Math.round(ciLevel * 100)}% CI`, value: `${ci[0]} – ${ci[1]}` });
        if (pValue != null) rows.push({ label: 'P-value', value: String(pValue) });
        if (testStat != null) rows.push({ label: 'Test Statistic', value: String(testStat) });
        if (df != null) rows.push({ label: 'Degrees of Freedom', value: String(df) });

        return `<div class="result-display">
            <div class="result-header">
                <h4 class="result-method">${esc(method)}</h4>
                ${statusBadge(execStatus, execStatus === 'success' ? 'Executed' : execStatus)}
            </div>
            <div class="result-values">
                ${rows.map(r => `<div class="result-row">
                    <span class="result-label">${esc(r.label)}</span>
                    <span class="result-value mono">${esc(r.value)}</span>
                </div>`).join('')}
            </div>
            <div class="result-defensibility">
                <span class="defensibility-label">Defensibility</span>
                <span class="badge badge-${STATIS_STATE.statusClass(defensibility === 'defensible' ? 'complete' : defensibility === 'defensible_with_caveats' ? 'warning' : defensibility === 'review_required' ? 'review' : defensibility === 'not_defensible' ? 'error' : 'unknown')}">
                    ${esc(defensibilityLabels[defensibility] || defensibility)}
                </span>
            </div>
            ${fingerprint ? `<div class="result-fingerprint">${expandable('Result Fingerprint', `<code>${esc(fingerprint)}</code>`)}</div>` : ''}
        </div>`;
    }

    /**
     * Diagnostics display.
     */
    function diagnosticsDisplay(diagnostics) {
        if (!diagnostics || !Object.keys(diagnostics).length) {
            return empty('No diagnostics available for this analysis.');
        }

        const items = Array.isArray(diagnostics) ? diagnostics : Object.entries(diagnostics).map(([k, v]) => ({ name: k, ...v }));

        return `<div class="diagnostics-display">
            <h4>Diagnostics</h4>
            <div class="diagnostics-list">
                ${items.map(d => {
                    const name = d.name || d.check || 'Diagnostic';
                    const status = d.status || d.result || 'not_checked';
                    const detail = d.detail || d.message || '';
                    return `<div class="diagnostic-item">
                        <span class="diagnostic-name">${esc(name)}</span>
                        ${statusBadge(status)}
                        ${detail ? `<span class="diagnostic-detail">${esc(detail)}</span>` : ''}
                    </div>`;
                }).join('')}
            </div>
        </div>`;
    }

    /**
     * Defensibility display with reasons.
     */
    function defensibilityDisplay(status, reasons = []) {
        if (!status) return empty('Defensibility has not been assessed for this analysis.');

        const labels = {
            'defensible': 'DEFENSIBLE',
            'defensible_with_caveats': 'DEFENSIBLE WITH CAVEATS',
            'review_required': 'REVIEW REQUIRED',
            'not_defensible': 'NOT DEFENSIBLE',
            'not_assessed': 'NOT ASSESSED',
        };

        const badgeCls = status === 'defensible' ? 'complete' :
            status === 'defensible_with_caveats' ? 'warning' :
            status === 'review_required' ? 'review' :
            status === 'not_defensible' ? 'error' : 'unknown';

        return `<div class="defensibility-display">
            <div class="defensibility-header">
                <h4>Analysis Defensibility</h4>
                <span class="badge badge-${badgeCls}" role="status">${esc(labels[status] || status)}</span>
            </div>
            ${reasons.length > 0 ? `<div class="defensibility-reasons">
                <span class="reasons-label">Reasons:</span>
                <ul>${reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>
            </div>` : ''}
            <p class="defensibility-note">This assessment reflects deterministic methodological checks. It does not indicate statistical significance, clinical importance, or publishability.</p>
        </div>`;
    }

    /**
     * Audit entry display.
     */
    function auditEntry(entry) {
        const time = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : 'Unknown time';
        return `<article class="audit-entry" role="article">
            <div class="audit-entry-header">
                <time class="audit-time" datetime="${esc(entry.timestamp || '')}">${esc(time)}</time>
                <span class="audit-action">${esc(entry.action || 'Unknown action')}</span>
                <span class="audit-agent">${esc(entry.agent || '')}</span>
            </div>
            <p class="audit-decision">${esc(entry.decision || '')}</p>
            ${entry.researcher_decision ? `<span class="audit-researcher-decision">Researcher: ${esc(entry.researcher_decision)}</span>` : ''}
            ${entry.confidence ? `<span class="audit-confidence">Confidence: ${esc(entry.confidence)}</span>` : ''}
        </article>`;
    }

    /**
     * Notification toast container.
     */
    function notificationContainer() {
        return `<div id="notifications" class="notifications" role="status" aria-live="polite" aria-atomic="false"></div>`;
    }

    /**
     * Render a single notification.
     */
    function notificationHtml(n) {
        return `<div class="notification notification-${esc(n.type)}" data-id="${n.id}">
            <span class="notification-message">${esc(n.message)}</span>
            <button class="notification-close" onclick="STATIS_STATE.dismissNotification(${n.id})" aria-label="Dismiss notification">×</button>
        </div>`;
    }

    /**
     * Format a date string for display.
     */
    function formatDate(isoString) {
        if (!isoString) return 'Unknown';
        try {
            const d = new Date(isoString);
            return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch {
            return isoString;
        }
    }

    /**
     * Format relative time.
     */
    function relativeTime(isoString) {
        if (!isoString) return 'Unknown';
        try {
            const d = new Date(isoString);
            const now = new Date();
            const diff = now - d;
            const mins = Math.floor(diff / 60000);
            if (mins < 1) return 'Just now';
            if (mins < 60) return `${mins} minute${mins !== 1 ? 's' : ''} ago`;
            const hours = Math.floor(mins / 60);
            if (hours < 24) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
            const days = Math.floor(hours / 24);
            if (days < 30) return `${days} day${days !== 1 ? 's' : ''} ago`;
            return formatDate(isoString);
        } catch {
            return isoString;
        }
    }

    /**
     * Truncate a hash for display.
     */
    function shortHash(hash, len = 12) {
        if (!hash) return '';
        return hash.length > len ? hash.substring(0, len) + '…' : hash;
    }

    return {
        esc,
        loading,
        empty,
        error,
        statusBadge,
        workflowStep,
        expandable,
        kvRow,
        cardHeader,
        actionButtons,
        projectCard,
        analysisCard,
        resultDisplay,
        diagnosticsDisplay,
        defensibilityDisplay,
        auditEntry,
        notificationContainer,
        notificationHtml,
        formatDate,
        relativeTime,
        shortHash,
    };
})();

if (typeof window !== 'undefined') window.STATIS_UI = STATIS_UI;
