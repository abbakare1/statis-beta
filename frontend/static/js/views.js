/**
 * STATIS Views
 * Researcher-facing view implementations.
 * Each view renders into #app-main and handles its own data loading.
 * Phase 2I: Frontend displays backend state; never fabricates or computes.
 */

const STATIS_VIEWS = (() => {
    'use strict';

    const API = () => window.STATIS_API;
    const UI = () => window.STATIS_UI;
    const STATE = () => window.STATIS_STATE;

    function el(id) { return document.getElementById(id); }
    function main() { return el('app-main'); }

    // =====================================================================
    // Dashboard View
    // =====================================================================
    async function dashboard() {
        STATE().set('currentView', 'dashboard');
        STATE().set('currentProjectId', null);
        el('nav-section')?.classList.remove('active');
        el('nav-project')?.classList.add('hidden');
        el('nav-project-name').textContent = '';

        main().innerHTML = `
            <div class="view-header">
                <div>
                    <h2 class="view-title">Dashboard</h2>
                    <p class="view-subtitle">Scientific Statistical Analysis and Research Intelligence Platform</p>
                </div>
                <button class="btn btn-primary" onclick="STATIS_VIEWS.showNewProjectForm()" aria-label="Create new research project">+ New Research Project</button>
            </div>

            <div id="new-project-form" class="card hidden">
                ${cardHeader('Create Research Project')}
                <form id="createProjectForm" class="project-form">
                    <div class="form-group">
                        <label for="projName">Project Name</label>
                        <input type="text" id="projName" name="name" required placeholder="e.g., GI Mitochondrial Study">
                    </div>
                    <div class="form-group">
                        <label for="projDesc">Brief Description</label>
                        <textarea id="projDesc" name="description" placeholder="A short description of this research project"></textarea>
                    </div>
                    <div class="form-group">
                        <label for="projRQ">Research Question</label>
                        <textarea id="projRQ" name="research_question" placeholder="What are you trying to determine?"></textarea>
                    </div>
                    <div class="form-group">
                        <label for="projDesign">Study Design</label>
                        <select id="projDesign" name="study_design">
                            <option value="">Select study design...</option>
                            <option value="cohort">Cohort</option>
                            <option value="rct">Randomized Controlled Trial (RCT)</option>
                            <option value="case-control">Case-control</option>
                            <option value="cross-sectional">Cross-sectional</option>
                            <option value="other">Other</option>
                            <option value="unsure">Unsure</option>
                        </select>
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary">Create Project</button>
                        <button type="button" class="btn btn-secondary" onclick="STATIS_VIEWS.hideNewProjectForm()">Cancel</button>
                    </div>
                </form>
            </div>

            <div class="section-header">
                <h3>My Research Projects</h3>
            </div>
            <div id="project-list" class="project-grid" role="list" aria-label="Research projects">
                ${UI().loading('Loading projects...')}
            </div>
        `;

        el('createProjectForm')?.addEventListener('submit', handleCreateProject);
        loadProjectList();
    }

    async function loadProjectList() {
        const listEl = el('project-list');
        if (!listEl) return;
        listEl.innerHTML = UI().loading('Loading projects...');
        STATE().startLoading('projects');

        try {
            const data = await API().projects.summary();
            STATE().set('projects', data.projects || []);
            STATE().setError('projects', null);

            const projects = data.projects || [];
            if (!projects.length) {
                listEl.innerHTML = UI().empty(
                    'No research projects yet.',
                    '<button class="btn btn-primary btn-sm" onclick="STATIS_VIEWS.showNewProjectForm()">Create your first project</button>'
                );
                return;
            }

            listEl.innerHTML = projects.map(p => UI().projectCard(p)).join('');
        } catch (err) {
            STATE().setError('projects', err);
            listEl.innerHTML = UI().error(UI() ? STATIS_API.humanizeError(err) : err.message, 'STATIS_VIEWS.loadProjectList()');
        } finally {
            STATE().endLoading('projects');
        }
    }

    function showNewProjectForm() {
        const form = el('new-project-form');
        if (form) {
            form.classList.remove('hidden');
            el('projName')?.focus();
        }
    }

    function hideNewProjectForm() {
        el('new-project-form')?.classList.add('hidden');
        el('createProjectForm')?.reset();
    }

    async function handleCreateProject(e) {
        e.preventDefault();
        const form = e.target;
        const btn = form.querySelector('button[type="submit"]');
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Creating...';

        try {
            const data = {
                name: el('projName').value.trim(),
                description: el('projDesc').value.trim(),
                research_question: el('projRQ').value.trim(),
                study_design: el('projDesign').value,
            };

            if (!data.name) {
                STATE().notify('Project name is required.', 'error');
                return;
            }

            const project = await API().projects.create(data);
            STATE().notify(`Project "${project.name}" created.`, 'success');
            hideNewProjectForm();
            STATIS_ROUTER.navigate(`/project/${project.id}`);
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }

    // =====================================================================
    // Project Workspace View
    // =====================================================================
    async function project(projectId) {
        STATE().set('currentView', 'project');
        STATE().set('currentProjectId', projectId);
        el('nav-section')?.classList.add('active');
        el('nav-project')?.classList.remove('hidden');
        el('nav-project-name').textContent = '';

        const subView = STATE().state.currentSubView || 'overview';

        main().innerHTML = `
            <div class="view-header">
                <div>
                    <nav class="breadcrumb" aria-label="Breadcrumb">
                        <a href="#/dashboard">Dashboard</a>
                        <span class="breadcrumb-sep" aria-hidden="true">/</span>
                        <span id="breadcrumb-project" class="breadcrumb-current">Project</span>
                    </nav>
                    <h2 class="view-title" id="project-title">Loading project...</h2>
                    <p class="view-subtitle" id="project-design"></p>
                </div>
            </div>

            <div class="project-workspace">
                <aside class="project-nav" aria-label="Project sections">
                    <nav id="project-nav-list">
                        ${projectNavItem('overview', 'Overview', subView)}
                        ${projectNavItem('pipeline', 'Pipeline', subView)}
                        ${projectNavItem('data', 'Data', subView)}
                        ${projectNavItem('analysis', 'Analysis', subView)}
                        ${projectNavItem('results', 'Results', subView)}
                        ${projectNavItem('outputs', 'Figures & Tables', subView)}
                        ${projectNavItem('reports', 'Reports', subView)}
                        ${projectNavItem('reproducibility', 'Reproducibility', subView)}
                        ${projectNavItem('audit', 'Audit', subView)}
                    </nav>
                </aside>
                <div class="project-content" id="project-content" role="main">
                    ${UI().loading('Loading project workspace...')}
                </div>
            </div>
        `;

        document.querySelectorAll('#project-nav-list .project-nav-item').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const sub = link.dataset.subview;
                STATE().set('currentSubView', sub);
                STATIS_ROUTER.navigate(`/project/${projectId}/${sub}`);
            });
        });

        await loadProjectWorkspace(projectId, subView);
    }

    function projectNavItem(sub, label, current) {
        const active = sub === current ? 'active' : '';
        return `<a href="#/project/${'' }/${sub}" class="project-nav-item ${active}" data-subview="${sub}" role="link" aria-current="${sub === current ? 'page' : 'false'}">${UI().esc(label)}</a>`;
    }

    async function loadProjectWorkspace(projectId, subView) {
        const content = el('project-content');
        if (!content) return;

        STATE().startLoading('project');
        content.innerHTML = UI().loading('Loading project workspace...');

        try {
            // Load project detail + workflow in parallel
            const [projectData, workflowData] = await Promise.all([
                API().projects.get(projectId),
                API().projects.workflow(projectId),
            ]);

            STATE().set('projectDetail', projectData);
            STATE().set('projectWorkflow', workflowData);

            // Update header
            el('project-title').textContent = projectData.name || 'Project';
            el('breadcrumb-project').textContent = projectData.name || 'Project';
            el('nav-project-name').textContent = projectData.name || '';
            el('project-design').textContent = projectData.study_design || '';

            // Render sub-view
            await renderProjectSubView(projectId, subView);
        } catch (err) {
            STATE().setError('project', err);
            content.innerHTML = UI().error(STATIS_API.humanizeError(err), `STATIS_VIEWS.project('${projectId}')`);
        } finally {
            STATE().endLoading('project');
        }
    }

    async function renderProjectSubView(projectId, subView) {
        const content = el('project-content');
        if (!content) return;

        const renderers = {
            'overview': renderProjectOverview,
            'pipeline': renderProjectPipeline,
            'data': renderProjectData,
            'analysis': renderProjectAnalysis,
            'results': renderProjectResults,
            'outputs': renderProjectOutputs,
            'reports': renderProjectReports,
            'reproducibility': renderProjectReproducibility,
            'audit': renderProjectAudit,
        };

        const renderer = renderers[subView] || renderProjectOverview;
        try {
            await renderer(projectId, content);
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    // ---------------------------------------------------------------------
    // Project Overview
    // ---------------------------------------------------------------------
    async function renderProjectOverview(projectId, content) {
        const workflow = STATE().state.projectWorkflow;
        const project = STATE().state.projectDetail;

        if (!workflow) {
            content.innerHTML = UI().error('Workflow data unavailable.');
            return;
        }

        const steps = workflow.steps || [];
        const reviewItems = workflow.review_items || [];
        const counts = workflow.counts || {};

        content.innerHTML = `
            <div class="overview-header">
                <h3>${UI().esc(project?.name || 'Project')}</h3>
                ${project?.study_design ? `<span class="badge badge-info">${UI().esc(project.study_design)}</span>` : ''}
            </div>

            <div class="overview-grid">
                <section class="overview-section">
                    <h4>Workflow Status</h4>
                    <div class="workflow-steps" role="list" aria-label="Project workflow steps">
                        ${steps.map(s => UI().workflowStep(s)).join('')}
                    </div>
                </section>

                <section class="overview-section">
                    <h4>Summary</h4>
                    <div class="overview-stats">
                        <div class="stat-item">
                            <span class="stat-value">${counts.datasets ?? 0}</span>
                            <span class="stat-label">Datasets</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${counts.analyses ?? 0}</span>
                            <span class="stat-label">Analyses</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${(counts.figures ?? 0) + (counts.tables ?? 0)}</span>
                            <span class="stat-label">Figures & Tables</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${counts.reports ?? 0}</span>
                            <span class="stat-label">Reports</span>
                        </div>
                    </div>
                </section>

                ${reviewItems.length > 0 ? `
                <section class="overview-section overview-review">
                    <h4>Requires Your Attention</h4>
                    <ul class="review-list">
                        ${reviewItems.map(r => `<li class="review-list-item">${UI().esc(r)}</li>`).join('')}
                    </ul>
                </section>` : ''}

                ${project?.research_question ? `
                <section class="overview-section">
                    <h4>Research Question</h4>
                    <p class="research-question">${UI().esc(project.research_question)}</p>
                </section>` : ''}
            </div>
        `;
    }

    // ---------------------------------------------------------------------
    // Project Pipeline
    // ---------------------------------------------------------------------
    async function renderProjectPipeline(projectId, content) {
        const workflow = STATE().state.projectWorkflow;
        const steps = workflow?.steps || [];

        // Map workflow steps to pipeline stages
        const pipelineStages = [
            { key: 'research_question', label: 'Research Question', desc: 'Define what you are trying to determine' },
            { key: 'study_design', label: 'Study Design', desc: 'Confirm the study design approach' },
            { key: 'dataset', label: 'Dataset', desc: 'Upload and register your dataset' },
            { key: 'data_quality', label: 'Data Quality', desc: 'Assess missingness, outliers, and data issues' },
            { key: 'variable_roles', label: 'Variable Roles', desc: 'Confirm outcome, exposure, and predictor roles' },
            { key: 'power', label: 'Power Assessment', desc: 'Review statistical power and sample size' },
            { key: 'analysis_plan', label: 'Analysis Plan', desc: 'Review proposed statistical analyses' },
            { key: 'researcher_review', label: 'Researcher Review', desc: 'Approve or modify proposed analyses' },
            { key: 'statistical_analysis', label: 'Statistical Analysis', desc: 'Execute approved analyses' },
            { key: 'diagnostics', label: 'Diagnostics', desc: 'Review statistical diagnostic checks' },
            { key: 'defensibility', label: 'Defensibility', desc: 'Review methodological defensibility' },
            { key: 'results', label: 'Results', desc: 'View statistical results' },
            { key: 'figures_tables', label: 'Figures & Tables', desc: 'Generate publication outputs' },
            { key: 'report', label: 'Report', desc: 'Generate structured report' },
            { key: 'reproducibility', label: 'Reproducibility', desc: 'Create and verify reproducibility package' },
        ];

        const stepMap = {};
        steps.forEach(s => { stepMap[s.step] = s; });

        content.innerHTML = `
            <div class="pipeline-header">
                <h3>Analysis Pipeline</h3>
                <p class="pipeline-subtitle">The full research workflow from question to reproducible outputs</p>
            </div>
            <div class="pipeline-flow" role="list" aria-label="Analysis pipeline stages">
                ${pipelineStages.map((stage, i) => {
                    const step = stepMap[stage.key];
                    const state = step?.state || 'not_started';
                    const label = STATE().statusLabel(state);
                    const cls = STATE().statusClass(state);
                    const icon = state === 'complete' ? '✓' : state === 'needs_review' ? '⚠' : state === 'error' ? '✗' : '○';
                    return `
                        <div class="pipeline-stage pipeline-${cls}" role="listitem">
                            <div class="pipeline-stage-number">${i + 1}</div>
                            <div class="pipeline-stage-icon" aria-hidden="true">${icon}</div>
                            <div class="pipeline-stage-info">
                                <span class="pipeline-stage-label">${UI().esc(stage.label)}</span>
                                <span class="pipeline-stage-status">${UI().esc(label)}</span>
                                <span class="pipeline-stage-desc">${UI().esc(stage.desc)}</span>
                            </div>
                        </div>
                        ${i < pipelineStages.length - 1 ? '<div class="pipeline-connector" aria-hidden="true">↓</div>' : ''}
                    `;
                }).join('')}
            </div>
        `;
    }

    // ---------------------------------------------------------------------
    // Project Data
    // ---------------------------------------------------------------------
    async function renderProjectData(projectId, content) {
        content.innerHTML = UI().loading('Loading datasets...');
        STATE().startLoading('datasets');

        try {
            const [dsData, qualityData, rolesData] = await Promise.all([
                API().datasets.list(projectId),
                API().projects.qualityProfile(projectId).catch(() => ({ profiles: [] })),
                API().projects.variableRoles(projectId).catch(() => ({ variable_roles: [] })),
            ]);

            STATE().set('datasets', dsData || []);
            STATE().set('qualityProfiles', qualityData?.profiles || []);
            STATE().set('variableRoles', rolesData?.variable_roles || []);

            const datasets = dsData || [];
            const profiles = qualityData?.profiles || [];
            const roles = rolesData?.variable_roles || [];

            const uploadFormHtml = `
                <div id="upload-form" class="card upload-card">
                    ${UI().cardHeader('Upload Dataset')}
                    <form id="uploadForm" enctype="multipart/form-data" class="upload-form">
                        <div class="form-group">
                            <label for="datasetFile">Dataset File</label>
                            <input type="file" id="datasetFile" accept=".csv,.tsv,.xlsx,.xls,.parquet" required>
                            <p class="form-hint">Supported formats: CSV, TSV, Excel (.xlsx/.xls), Parquet. Maximum ${500} MB.</p>
                        </div>
                        <div id="upload-status" class="upload-status hidden"></div>
                        <div class="form-actions">
                            <button type="submit" class="btn btn-primary" id="uploadBtn">Upload Dataset</button>
                        </div>
                    </form>
                </div>
            `;

            if (!datasets.length) {
                content.innerHTML = `
                    <div class="data-header"><h3>Data</h3></div>
                    ${uploadFormHtml}
                    <div class="dataset-list">
                        ${UI().empty('No datasets uploaded yet. Upload a CSV, TSV, Excel, or Parquet file to begin.')}
                    </div>
                `;
                el('uploadForm')?.addEventListener('submit', handleUpload);
                return;
            }

            content.innerHTML = `
                <div class="data-header"><h3>Data</h3></div>
                ${uploadFormHtml}
                <div class="dataset-list">
                    ${datasets.map(d => datasetCard(d, profiles, roles, projectId)).join('')}
                </div>
            `;
            el('uploadForm')?.addEventListener('submit', handleUpload);

            // Load preview for each dataset card
            datasets.forEach(d => loadDatasetPreview(d.id));
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        } finally {
            STATE().endLoading('datasets');
        }
    }

    async function loadDatasetPreview(datasetId) {
        const el2 = el(`preview-${datasetId}`);
        if (!el2) return;
        try {
            const preview = await API().datasets.preview(datasetId, 5);
            if (!preview || preview.error) {
                el2.innerHTML = `<p class="text-dim">Preview unavailable.</p>`;
                return;
            }
            const cols = (preview.columns || []).map(c =>
                `<span class="badge badge-info" title="${c.n_missing} missing">${UI().esc(c.name)} <span class="badge-dim">${UI().esc(c.dtype)}</span></span>`
            ).join(' ');
            const headRows = preview.head || [];
            const tableHtml = headRows.length ? `
                <div class="preview-table-wrap">
                    <table class="preview-table" role="table">
                        <thead><tr>${(headRows[0] ? Object.keys(headRows[0]) : []).map(k => `<th>${UI().esc(k)}</th>`).join('')}</tr></thead>
                        <tbody>
                            ${headRows.map(row => `<tr>${Object.values(row).map(v => `<td>${v == null ? '<span class="text-dim">—</span>' : UI().esc(String(v))}</td>`).join('')}</tr>`).join('')}
                        </tbody>
                    </table>
                </div>
            ` : '<p class="text-dim">No rows to display.</p>';

            el2.innerHTML = `
                <div class="preview-meta">
                    ${UI().kvRow('Rows', (preview.n_rows || 0).toLocaleString())}
                    ${UI().kvRow('Columns', preview.n_columns || 0)}
                    ${UI().kvRow('File size', formatBytes(preview.file_size_bytes))}
                </div>
                <div class="preview-columns">${cols}</div>
                ${tableHtml}
            `;
        } catch {
            el2.innerHTML = `<p class="text-dim">Preview unavailable.</p>`;
        }
    }

    function formatBytes(bytes) {
        if (bytes == null) return '';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    function datasetCard(dataset, profiles, roles, projectId) {
        const profile = profiles.find(p => p.dataset_id === dataset.id);
        const rowCount = dataset.row_count;
        const colCount = dataset.column_count;
        const hash = dataset.content_hash ? UI().shortHash(dataset.content_hash) : '';
        const cardProjectId = projectId || dataset.project_id || '';

        const qualitySummary = profile?.profile ? `
            <div class="dataset-quality-summary">
                ${UI().kvRow('Issues', profile.profile.n_issues ?? 0)}
                ${UI().kvRow('Critical', profile.profile.n_critical ?? 0)}
                ${UI().kvRow('Warnings', profile.profile.n_warning ?? 0)}
            </div>
        ` : '';

        return `<article class="dataset-card card" role="article">
            ${UI().cardHeader(
                dataset.original_filename || dataset.filename || 'Dataset',
                `<span class="badge badge-info">v${dataset.version ?? 1}</span>`,
                `Uploaded ${UI().formatDate(dataset.created_at)}`
            )}
            <div class="dataset-card-body">
                <div class="dataset-stats">
                    ${rowCount != null ? `<div class="stat-item"><span class="stat-value">${rowCount.toLocaleString()}</span><span class="stat-label">Observations</span></div>` : ''}
                    ${colCount != null ? `<div class="stat-item"><span class="stat-value">${colCount}</span><span class="stat-label">Variables</span></div>` : ''}
                </div>
                ${hash ? `<div class="dataset-hash">${UI().expandable('Dataset Identity', UI().kvRow('Content Hash', dataset.content_hash, true))}</div>` : ''}
                ${qualitySummary}
                <div class="dataset-preview" id="preview-${UI().esc(dataset.id)}">
                    ${UI().loading('Loading preview...')}
                </div>
            </div>
            <div class="dataset-card-footer">
                ${UI().actionButtons([
                    { label: 'Run Data Audit', onclick: `STATIS_VIEWS.runDataQuality('${UI().esc(dataset.id)}', '${UI().esc(dataset.project_id || '')}')`, primary: true },
                    { label: 'Go to Analysis', onclick: `STATIS_ROUTER.navigate('/project/${UI().esc(dataset.project_id || '')}/analysis')` },
                ])}
            </div>
        </article>`;
    }

    function showUploadForm(projectId) {
        el('upload-form')?.classList.remove('hidden');
        el('datasetFile')?.focus();
    }

    function hideUploadForm() {
        el('upload-form')?.classList.add('hidden');
    }

    function setUploadStatus(message, type) {
        const statusEl = el('upload-status');
        if (!statusEl) return;
        statusEl.className = `upload-status upload-status-${type}`;
        statusEl.textContent = message;
        statusEl.classList.remove('hidden');
    }

    function clearUploadStatus() {
        const statusEl = el('upload-status');
        if (!statusEl) return;
        statusEl.className = 'upload-status hidden';
        statusEl.textContent = '';
    }

    async function handleUpload(e) {
        e.preventDefault();
        const file = el('datasetFile')?.files[0];
        if (!file) {
            setUploadStatus('Please select a file to upload.', 'error');
            return;
        }

        const projectId = STATE().state.currentProjectId;
        if (!projectId) {
            setUploadStatus('No project context found. Open a project first.', 'error');
            return;
        }

        const btn = el('uploadBtn') || e.target.querySelector('button[type="submit"]');
        btn.disabled = true;
        btn.textContent = 'Uploading...';
        clearUploadStatus();
        setUploadStatus(`Uploading ${file.name}...`, 'info');

        try {
            const result = await API().datasets.upload(projectId, file);
            const rows = result.row_count != null ? result.row_count.toLocaleString() : '?';
            const cols = result.column_count != null ? result.column_count : '?';
            setUploadStatus(`Uploaded successfully — ${rows} rows, ${cols} columns.`, 'success');
            STATE().notify(`Dataset uploaded: ${rows} rows × ${cols} columns.`, 'success');

            // Reset form
            if (el('datasetFile')) el('datasetFile').value = '';

            // Re-render Data view after brief delay so user sees success message
            setTimeout(() => {
                renderProjectData(projectId, el('project-content'));
            }, 600);
        } catch (err) {
            const msg = STATIS_API.humanizeError(err);
            setUploadStatus(msg, 'error');
            STATE().notify(msg, 'error');
            btn.disabled = false;
            btn.textContent = 'Upload Dataset';
        }
    }

    async function runDataQuality(datasetId, projectId) {
        STATE().notify('Running data quality assessment...', 'info');
        try {
            const result = await API().quality.assess(datasetId, {});
            STATE().notify(`Data quality assessment complete: ${result.n_issues ?? 0} issues found.`, 'success');
            renderProjectData(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    // ---------------------------------------------------------------------
    // Project Analysis
    // ---------------------------------------------------------------------
    async function renderProjectAnalysis(projectId, content) {
        content.innerHTML = UI().loading('Loading analysis plans...');
        STATE().startLoading('plans');

        try {
            const [plansData, rolesData] = await Promise.all([
                API().projects.plans(projectId),
                API().projects.variableRoles(projectId).catch(() => ({ variable_roles: [] })),
            ]);

            STATE().set('plans', plansData?.plans || []);
            STATE().set('variableRoles', rolesData?.variable_roles || []);

            const plans = plansData?.plans || [];
            const roles = rolesData?.variable_roles || [];

            let plansHtml = '';
            if (!plans.length) {
                plansHtml = UI().empty(
                    'No analysis plans yet.',
                    `<button class="btn btn-primary btn-sm" onclick="STATIS_VIEWS.generateAnalysisPlan('${UI().esc(projectId)}')">Generate Analysis Plan</button>`
                );
            } else {
                // For each plan, load its analyses
                const planDetails = await Promise.all(
                    plans.map(p => API().analysisPlan.get(p.plan_id).catch(() => null))
                );
                plansHtml = planDetails.filter(Boolean).map(plan => `
                    <section class="plan-section">
                        ${UI().cardHeader(`Plan ${plan.plan_id?.substring(0, 8) || ''}`, '', `${plan.analyses?.length || 0} analyses · ${UI().formatDate(plan.created_at)}`)}
                        <div class="analysis-cards">
                            ${(plan.analyses || []).map(a => UI().analysisCard(a, plan.plan_id)).join('')}
                        </div>
                    </section>
                `).join('');
            }

            const rolesHtml = roles.length > 0 ? `
                <section class="roles-section">
                    ${UI().cardHeader('Variable Roles')}
                    <div class="roles-list">
                        ${roles.map(r => `
                            <div class="role-item">
                                <span class="role-variable mono">${UI().esc(r.variable_name)}</span>
                                <span class="badge badge-${r.status === 'resolved' ? 'complete' : r.requires_researcher_review ? 'review' : 'draft'}">${UI().esc(r.proposed_role)}</span>
                                <span class="role-source">${UI().esc(r.role_source)}</span>
                                ${r.requires_researcher_review ? '<span class="badge badge-review">Needs Review</span>' : ''}
                            </div>
                        `).join('')}
                    </div>
                </section>
            ` : '';

            content.innerHTML = `
                <div class="analysis-header">
                    <h3>Analysis Plan</h3>
                    <button class="btn btn-primary btn-sm" onclick="STATIS_VIEWS.generateAnalysisPlan('${UI().esc(projectId)}')">Generate New Plan</button>
                </div>
                ${plansHtml}
                ${rolesHtml}
            `;
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        } finally {
            STATE().endLoading('plans');
        }
    }

    async function generateAnalysisPlan(projectId) {
        const project = STATE().state.projectDetail;
        STATE().notify('Generating analysis plan...', 'info');

        try {
            const datasets = await API().datasets.list(projectId);
            const datasetId = datasets?.[0]?.id || null;

            const result = await API().analysisPlan.create({
                project_id: projectId,
                dataset_id: datasetId,
                research_question: project?.research_question || '',
                study_design: project?.study_design || '',
            });

            STATE().notify(`Analysis plan created with ${result.n_analyses ?? 0} proposed analyses.`, 'success');
            renderProjectAnalysis(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    async function reviewAnalysis(planId, analysisId) {
        if (!planId || !analysisId) {
            STATE().notify('Analysis details unavailable.', 'error');
            return;
        }

        STATE().notify('Loading analysis details...', 'info');
        try {
            const plan = await API().analysisPlan.get(planId);
            const analysis = (plan.analyses || []).find(a => a.analysis_id === analysisId);
            if (!analysis) {
                STATE().notify('Analysis not found in plan.', 'error');
                return;
            }

            // Show review modal
            showAnalysisReviewModal(plan, analysis);
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    function showAnalysisReviewModal(plan, analysis) {
        const modal = el('modal-overlay');
        if (!modal) return;

        const validation = analysis.validation || 'pending';
        const approval = analysis.approval || 'proposed';
        const canApprove = validation !== 'block' && approval !== 'approved';

        modal.innerHTML = `
            <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
                <div class="modal-header">
                    <h3 id="modal-title">${UI().esc(analysis.analysis_name || analysis.method || 'Analysis Review')}</h3>
                    <button class="modal-close" onclick="STATIS_VIEWS.closeModal()" aria-label="Close dialog">×</button>
                </div>
                <div class="modal-body">
                    <div class="review-section">
                        <h4>Specification</h4>
                        ${UI().kvRow('Method', analysis.method)}
                        ${UI().kvRow('Outcome', analysis.outcome)}
                        ${(analysis.predictors || []).length ? UI().kvRow('Predictors', analysis.predictors.join(', ')) : ''}
                        ${(analysis.covariates || []).length ? UI().kvRow('Covariates', analysis.covariates.join(', ')) : ''}
                        ${UI().kvRow('Category', analysis.analysis_class)}
                        ${UI().kvRow('Prespecified', analysis.prespecified)}
                    </div>
                    <div class="review-section">
                        <h4>Validation</h4>
                        ${UI().kvRow('Status', validation === 'passed' ? 'Passed' : validation === 'block' ? 'Blocked' : validation === 'warning' ? 'Warning' : 'Pending')}
                        ${(analysis.validation_blockers || []).length ? `
                            <div class="review-concerns">
                                <strong>Blockers:</strong>
                                <ul>${analysis.validation_blockers.map(b => `<li>${UI().esc(b)}</li>`).join('')}</ul>
                            </div>` : ''}
                        ${(analysis.warnings || []).length ? `
                            <div class="review-concerns">
                                <strong>Warnings:</strong>
                                <ul>${analysis.warnings.map(w => `<li>${UI().esc(w)}</li>`).join('')}</ul>
                            </div>` : ''}
                    </div>
                    ${analysis.rationale ? `
                    <div class="review-section">
                        <h4>Why STATIS Proposed This</h4>
                        <p>${UI().esc(analysis.rationale)}</p>
                    </div>` : ''}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="STATIS_VIEWS.closeModal()">Close</button>
                    <button class="btn btn-primary" onclick="STATIS_VIEWS.approveAnalysis('${UI().esc(plan.plan_id)}', '${UI().esc(analysis.analysis_id)}')" ${canApprove ? '' : 'disabled'}>Approve Analysis</button>
                </div>
            </div>
        `;
        modal.classList.remove('hidden');
    }

    function closeModal() {
        el('modal-overlay')?.classList.add('hidden');
        if (el('modal-overlay')) el('modal-overlay').innerHTML = '';
    }

    async function approveAnalysis(planId, analysisId) {
        STATE().notify('Approving analysis...', 'info');
        try {
            await API().analysisPlan.approve(analysisId, planId);
            STATE().notify('Analysis approved.', 'success');
            closeModal();
            const projectId = STATE().state.currentProjectId;
            if (projectId) renderProjectAnalysis(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    async function executeAnalysis(planId, analysisId) {
        STATE().notify('Executing analysis...', 'info');
        try {
            const result = await API().analysisPlan.execute(analysisId, planId);
            if (result.success) {
                STATE().notify('Analysis executed successfully.', 'success');
            } else {
                STATE().notify('Analysis execution failed: ' + (result.data?.error || 'Unknown error'), 'error');
            }
            const projectId = STATE().state.currentProjectId;
            if (projectId) renderProjectAnalysis(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    // ---------------------------------------------------------------------
    // Project Results
    // ---------------------------------------------------------------------
    async function renderProjectResults(projectId, content) {
        content.innerHTML = UI().loading('Loading analysis results...');
        STATE().startLoading('runs');

        try {
            const runsData = await API().projects.runs(projectId);
            const runs = runsData?.runs || [];
            STATE().set('runs', runs);

            if (!runs.length) {
                content.innerHTML = `
                    <div class="results-header"><h3>Results</h3></div>
                    ${UI().empty('No analyses have been executed yet.', `<a href="#/project/${UI().esc(projectId)}/analysis" class="btn btn-primary btn-sm">Go to Analysis Plan</a>`)}
                `;
                return;
            }

            // Load full run details for each
            const runDetails = await Promise.all(
                runs.map(r => API().runs.get(r.id).catch(() => r))
            );

            content.innerHTML = `
                <div class="results-header"><h3>Results</h3></div>
                <div class="results-list">
                    ${runDetails.map(run => resultCard(run)).join('')}
                </div>
            `;
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        } finally {
            STATE().endLoading('runs');
        }
    }

    function resultCard(run) {
        let resultData = {};
        try {
            resultData = run.result_json ? JSON.parse(run.result_json) : (run.result || {});
        } catch { resultData = {}; }

        const method = run.analysis_type || resultData.method_name || 'Unknown';
        const success = run.success !== false;
        const defensibility = resultData.defensibility_status || 'not_assessed';

        return `<article class="result-card card" role="article">
            ${UI().cardHeader(method, UI().statusBadge(success ? 'executed' : 'validation_failed'), `Run ${UI().shortHash(run.id)}`)}
            <div class="result-card-body">
                ${UI().resultDisplay(resultData)}
                ${resultData.diagnostic_summary ? UI().diagnosticsDisplay(resultData.diagnostic_summary) : ''}
                ${UI().defensibilityDisplay(defensibility, resultData.methodological_warnings || resultData.review_required || [])}
            </div>
            <div class="result-card-footer">
                ${UI().actionButtons([
                    { label: 'View Provenance', onclick: `STATIS_VIEWS.showProvenance('${UI().esc(run.id)}')` },
                ])}
            </div>
        </article>`;
    }

    async function showProvenance(runId) {
        STATE().notify('Loading provenance...', 'info');
        try {
            const run = await API().runs.get(runId);
            const modal = el('modal-overlay');
            if (!modal) return;

            modal.innerHTML = `
                <div class="modal" role="dialog" aria-modal="true" aria-labelledby="prov-title">
                    <div class="modal-header">
                        <h3 id="prov-title">Provenance</h3>
                        <button class="modal-close" onclick="STATIS_VIEWS.closeModal()" aria-label="Close dialog">×</button>
                    </div>
                    <div class="modal-body">
                        <div class="review-section">
                            <h4>Trace</h4>
                            ${UI().kvRow('Project', run.project_id)}
                            ${UI().kvRow('Dataset', run.dataset_id)}
                            ${UI().kvRow('Analysis Type', run.analysis_type)}
                            ${UI().kvRow('Input Fingerprint', run.input_fingerprint, true)}
                            ${UI().kvRow('Result Fingerprint', run.result_fingerprint, true)}
                            ${UI().kvRow('Engine', run.engine_version)}
                            ${UI().kvRow('Python', run.python_version)}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="STATIS_VIEWS.closeModal()">Close</button>
                    </div>
                </div>
            `;
            modal.classList.remove('hidden');
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    // ---------------------------------------------------------------------
    // Project Outputs (Figures & Tables)
    // ---------------------------------------------------------------------
    async function renderProjectOutputs(projectId, content) {
        content.innerHTML = UI().loading('Loading outputs...');
        STATE().startLoading('outputs');

        try {
            const outputsData = await API().projects.outputs(projectId);
            STATE().set('outputs', outputsData);

            const figures = outputsData?.figures || [];
            const tables = outputsData?.tables || [];

            if (!figures.length && !tables.length) {
                content.innerHTML = `
                    <div class="outputs-header"><h3>Figures & Tables</h3></div>
                    ${UI().empty('No publication outputs yet. Outputs are generated after analyses are executed.', `<a href="#/project/${UI().esc(projectId)}/results" class="btn btn-secondary btn-sm">View Results</a>`)}
                `;
                return;
            }

            content.innerHTML = `
                <div class="outputs-header"><h3>Figures & Tables</h3></div>
                ${figures.length ? `
                    <section class="outputs-section">
                        <h4>Figures (${figures.length})</h4>
                        <div class="outputs-grid">
                            ${figures.map(f => outputCard(f, 'figure', projectId)).join('')}
                        </div>
                    </section>` : ''}
                ${tables.length ? `
                    <section class="outputs-section">
                        <h4>Tables (${tables.length})</h4>
                        <div class="outputs-grid">
                            ${tables.map(t => outputCard(t, 'table', projectId)).join('')}
                        </div>
                    </section>` : ''}
            `;
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        } finally {
            STATE().endLoading('outputs');
        }
    }

    function outputCard(output, type, projectId) {
        const typeName = type === 'figure' ? (output.figure_type || 'Figure') : (output.table_type || 'Table');
        const lifecycle = output.lifecycle || 'draft';

        const actions = [
            { label: 'Validate', onclick: `STATIS_VIEWS.validateOutput('${type}', '${UI().esc(output.id)}', '${UI().esc(projectId)}')`, hidden: lifecycle === 'approved' || lifecycle === 'archived' },
            { label: 'Render', onclick: `STATIS_VIEWS.renderOutput('${type}', '${UI().esc(output.id)}', '${UI().esc(projectId)}')`, hidden: lifecycle === 'approved' || lifecycle === 'archived' },
            { label: 'Approve', onclick: `STATIS_VIEWS.approveOutput('${type}', '${UI().esc(output.id)}', '${UI().esc(projectId)}')`, disabled: lifecycle !== 'generated' && lifecycle !== 'review_required', hidden: lifecycle === 'approved' || lifecycle === 'archived' },
            { label: 'Download', onclick: `STATIS_VIEWS.downloadOutput('${type}', '${UI().esc(output.id)}')`, hidden: !output.artifact_format },
        ];

        return `<article class="output-card card" role="article">
            ${UI().cardHeader(typeName, UI().statusBadge(lifecycle), type === 'figure' ? 'Figure' : 'Table')}
            <div class="output-card-body">
                ${UI().kvRow('Classification', output.analytical_classification)}
                ${UI().kvRow('Prespecified', output.prespecified_status)}
                ${output.artifact_format ? UI().kvRow('Format', output.artifact_format.toUpperCase()) : ''}
            </div>
            <div class="output-card-footer">
                ${UI().actionButtons(actions)}
            </div>
        </article>`;
    }

    async function validateOutput(type, outputId, projectId) {
        STATE().notify('Validating output...', 'info');
        try {
            const api = type === 'figure' ? API().figures : API().tables;
            await api.validate(outputId);
            STATE().notify('Output validated.', 'success');
            renderProjectOutputs(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    async function renderOutput(type, outputId, projectId) {
        STATE().notify('Rendering output...', 'info');
        try {
            const api = type === 'figure' ? API().figures : API().tables;
            await api.render(outputId);
            STATE().notify('Output rendered.', 'success');
            renderProjectOutputs(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    async function approveOutput(type, outputId, projectId) {
        STATE().notify('Approving output...', 'info');
        try {
            const api = type === 'figure' ? API().figures : API().tables;
            await api.approve(outputId, {});
            STATE().notify('Output approved.', 'success');
            renderProjectOutputs(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    function downloadOutput(type, outputId) {
        const api = type === 'figure' ? API().figures : API().tables;
        window.open(api.downloadUrl(outputId), '_blank');
    }

    // ---------------------------------------------------------------------
    // Project Reports
    // ---------------------------------------------------------------------
    async function renderProjectReports(projectId, content) {
        content.innerHTML = UI().loading('Loading reports...');
        try {
            const outputsData = await API().projects.outputs(projectId);
            const reports = outputsData?.reports || [];

            if (!reports.length) {
                content.innerHTML = `
                    <div class="reports-header"><h3>Reports</h3></div>
                    ${UI().empty('No reports generated yet.', `<a href="#/project/${UI().esc(projectId)}/outputs" class="btn btn-secondary btn-sm">View Figures & Tables</a>`)}
                `;
                return;
            }

            content.innerHTML = `
                <div class="reports-header"><h3>Reports</h3></div>
                <div class="reports-list">
                    ${reports.map(r => `
                        <article class="report-card card" role="article">
                            ${UI().cardHeader(r.title || 'Report', UI().statusBadge(r.lifecycle), `Created ${UI().formatDate(r.created_at)}`)}
                            <div class="report-card-body">
                                ${UI().kvRow('Classification', r.analytical_classification)}
                                ${UI().kvRow('Format', r.artifact_format)}
                            </div>
                            <div class="report-card-footer">
                                ${UI().actionButtons([
                                    { label: 'Approve', onclick: `STATIS_VIEWS.approveOutput('report', '${UI().esc(r.id)}', '${UI().esc(projectId)}')`, disabled: r.lifecycle !== 'generated' && r.lifecycle !== 'review_required' },
                                    { label: 'Download', onclick: `STATIS_VIEWS.downloadOutput('report', '${UI().esc(r.id)}')`, hidden: !r.artifact_format },
                                ])}
                            </div>
                        </article>
                    `).join('')}
                </div>
            `;
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    // ---------------------------------------------------------------------
    // Project Reproducibility
    // ---------------------------------------------------------------------
    async function renderProjectReproducibility(projectId, content) {
        content.innerHTML = UI().loading('Loading reproducibility packages...');
        try {
            const reproData = await API().projects.reproducibility(projectId);
            const packages = reproData?.packages || [];

            if (!packages.length) {
                content.innerHTML = `
                    <div class="repro-header"><h3>Reproducibility</h3></div>
                    ${UI().empty('No reproducibility packages yet.', `<button class="btn btn-primary btn-sm" onclick="STATIS_VIEWS.createReproPackage('${UI().esc(projectId)}')">Create Package</button>`)}
                `;
                return;
            }

            content.innerHTML = `
                <div class="repro-header">
                    <h3>Reproducibility</h3>
                    <button class="btn btn-primary btn-sm" onclick="STATIS_VIEWS.createReproPackage('${UI().esc(projectId)}')">Create Package</button>
                </div>
                <div class="repro-list">
                    ${packages.map(p => `
                        <article class="repro-card card" role="article">
                            ${UI().cardHeader(p.package_id || 'Package', UI().statusBadge(p.verified ? 'complete' : 'draft', p.verified ? 'Verified' : 'Unverified'), `Created ${UI().formatDate(p.created_at)}`)}
                            <div class="repro-card-body">
                                ${UI().kvRow('Manifest Hash', p.manifest_hash ? UI().shortHash(p.manifest_hash) : '', true)}
                                ${UI().kvRow('Includes Raw Data', p.includes_raw_data ? 'Yes' : 'No')}
                            </div>
                            <div class="repro-card-footer">
                                ${UI().actionButtons([
                                    { label: 'Verify', onclick: `STATIS_VIEWS.verifyReproPackage('${UI().esc(p.package_id)}', '${UI().esc(projectId)}')`, primary: !p.verified },
                                    { label: 'View Manifest', onclick: `STATIS_VIEWS.viewReproManifest('${UI().esc(p.package_id)}')` },
                                ])}
                            </div>
                        </article>
                    `).join('')}
                </div>
            `;
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    async function createReproPackage(projectId) {
        STATE().notify('Creating reproducibility package...', 'info');
        try {
            const runsData = await API().projects.runs(projectId);
            const runIds = (runsData?.runs || []).map(r => r.id);

            const outputsData = await API().projects.outputs(projectId);
            const outputIds = [
                ...(outputsData?.figures || []).map(f => f.id),
                ...(outputsData?.tables || []).map(t => t.id),
            ];

            await API().reproducibility.create({
                project_id: projectId,
                analysis_run_ids: runIds,
                output_ids: outputIds,
            });
            STATE().notify('Reproducibility package created.', 'success');
            renderProjectReproducibility(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    async function verifyReproPackage(packageId, projectId) {
        STATE().notify('Verifying reproducibility package...', 'info');
        try {
            const result = await API().reproducibility.verify(packageId, {});
            if (result.verified) {
                STATE().notify('Package verified successfully.', 'success');
            } else {
                STATE().notify('Package verification failed. ' + (result.message || ''), 'error');
            }
            renderProjectReproducibility(projectId, el('project-content'));
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    async function viewReproManifest(packageId) {
        STATE().notify('Loading manifest...', 'info');
        try {
            const result = await API().reproducibility.manifest(packageId);
            const modal = el('modal-overlay');
            if (!modal) return;

            const manifest = result.manifest || result;
            modal.innerHTML = `
                <div class="modal modal-lg" role="dialog" aria-modal="true" aria-labelledby="manifest-title">
                    <div class="modal-header">
                        <h3 id="manifest-title">Reproducibility Manifest</h3>
                        <button class="modal-close" onclick="STATIS_VIEWS.closeModal()" aria-label="Close dialog">×</button>
                    </div>
                    <div class="modal-body">
                        <pre class="manifest-json">${UI().esc(JSON.stringify(manifest, null, 2))}</pre>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="STATIS_VIEWS.closeModal()">Close</button>
                    </div>
                </div>
            `;
            modal.classList.remove('hidden');
        } catch (err) {
            STATE().notify(STATIS_API.humanizeError(err), 'error');
        }
    }

    // ---------------------------------------------------------------------
    // Project Audit
    // ---------------------------------------------------------------------
    async function renderProjectAudit(projectId, content) {
        content.innerHTML = UI().loading('Loading audit history...');
        try {
            const auditData = await API().audit.list({ project_id: projectId, limit: 100 });
            const entries = auditData || [];

            if (!entries.length) {
                content.innerHTML = `
                    <div class="audit-header"><h3>Audit History</h3></div>
                    ${UI().empty('No audit entries yet for this project.')}
                `;
                return;
            }

            content.innerHTML = `
                <div class="audit-header"><h3>Audit History</h3></div>
                <div class="audit-list" role="list" aria-label="Audit history">
                    ${entries.map(e => UI().auditEntry(e)).join('')}
                </div>
            `;
        } catch (err) {
            content.innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    // =====================================================================
    // Global Views
    // =====================================================================
    async function resultsGlobal() {
        STATE().set('currentView', 'results');
        main().innerHTML = `
            <div class="view-header">
                <h2 class="view-title">Results</h2>
                <p class="view-subtitle">Statistical results across all projects</p>
            </div>
            <div id="global-results" class="results-list">
                ${UI().loading('Loading results...')}
            </div>
        `;

        try {
            const runsData = await API().runs.list({ limit: 50 });
            const runs = runsData?.runs || runsData || [];

            if (!runs.length) {
                el('global-results').innerHTML = UI().empty('No analysis results yet.', '<a href="#/dashboard" class="btn btn-primary btn-sm">Go to Dashboard</a>');
                return;
            }

            const runDetails = await Promise.all(runs.slice(0, 20).map(r => API().runs.get(r.id).catch(() => r)));
            el('global-results').innerHTML = runDetails.map(r => resultCard(r)).join('');
        } catch (err) {
            el('global-results').innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    async function outputsGlobal() {
        STATE().set('currentView', 'outputs');
        main().innerHTML = `
            <div class="view-header">
                <h2 class="view-title">Figures & Tables</h2>
                <p class="view-subtitle">Publication outputs across all projects</p>
            </div>
            <div id="global-outputs">
                ${UI().loading('Loading outputs...')}
            </div>
        `;

        try {
            const [figData, tblData] = await Promise.all([
                API().figures.list({ limit: 50 }),
                API().tables.list({ limit: 50 }),
            ]);

            const figures = figData?.outputs || [];
            const tables = tblData?.outputs || [];

            if (!figures.length && !tables.length) {
                el('global-outputs').innerHTML = UI().empty('No publication outputs yet.');
                return;
            }

            el('global-outputs').innerHTML = `
                ${figures.length ? `
                    <section class="outputs-section">
                        <h4>Figures (${figures.length})</h4>
                        <div class="outputs-grid">
                            ${figures.map(f => `
                                <article class="output-card card">
                                    ${UI().cardHeader(f.figure_type || 'Figure', UI().statusBadge(f.lifecycle), f.analytical_classification)}
                                    <div class="output-card-footer">
                                        ${UI().actionButtons([
                                            { label: 'Download', onclick: `STATIS_VIEWS.downloadOutput('figure', '${UI().esc(f.id)}')`, hidden: !f.artifact_format },
                                        ])}
                                    </div>
                                </article>
                            `).join('')}
                        </div>
                    </section>` : ''}
                ${tables.length ? `
                    <section class="outputs-section">
                        <h4>Tables (${tables.length})</h4>
                        <div class="outputs-grid">
                            ${tables.map(t => `
                                <article class="output-card card">
                                    ${UI().cardHeader(t.table_type || 'Table', UI().statusBadge(t.lifecycle), t.analytical_classification)}
                                    <div class="output-card-footer">
                                        ${UI().actionButtons([
                                            { label: 'Download', onclick: `STATIS_VIEWS.downloadOutput('table', '${UI().esc(t.id)}')`, hidden: !t.artifact_format },
                                        ])}
                                    </div>
                                </article>
                            `).join('')}
                        </div>
                    </section>` : ''}
            `;
        } catch (err) {
            el('global-outputs').innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    async function reportsGlobal() {
        STATE().set('currentView', 'reports');
        main().innerHTML = `
            <div class="view-header">
                <h2 class="view-title">Reports</h2>
                <p class="view-subtitle">Structured reports across all projects</p>
            </div>
            <div id="global-reports">
                ${UI().loading('Loading reports...')}
            </div>
        `;

        try {
            const repData = await API().reports.list({ limit: 50 });
            const reports = repData?.outputs || [];

            if (!reports.length) {
                el('global-reports').innerHTML = UI().empty('No reports yet.');
                return;
            }

            el('global-reports').innerHTML = `
                <div class="reports-list">
                    ${reports.map(r => `
                        <article class="report-card card">
                            ${UI().cardHeader(r.title || 'Report', UI().statusBadge(r.lifecycle), r.analytical_classification)}
                            <div class="report-card-footer">
                                ${UI().actionButtons([
                                    { label: 'Download', onclick: `STATIS_VIEWS.downloadOutput('report', '${UI().esc(r.id)}')`, hidden: !r.artifact_format },
                                ])}
                            </div>
                        </article>
                    `).join('')}
                </div>
            `;
        } catch (err) {
            el('global-reports').innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    async function auditGlobal() {
        STATE().set('currentView', 'audit');
        main().innerHTML = `
            <div class="view-header">
                <h2 class="view-title">Audit Ledger</h2>
                <p class="view-subtitle">Immutable record of every analytical decision</p>
            </div>
            <div id="global-audit" class="audit-list" role="list" aria-label="Audit history">
                ${UI().loading('Loading audit history...')}
            </div>
        `;

        try {
            const entries = await API().audit.list({ limit: 200 });
            if (!entries.length) {
                el('global-audit').innerHTML = UI().empty('No audit entries yet.');
                return;
            }
            el('global-audit').innerHTML = entries.map(e => UI().auditEntry(e)).join('');
        } catch (err) {
            el('global-audit').innerHTML = UI().error(STATIS_API.humanizeError(err));
        }
    }

    // =====================================================================
    // Helpers
    // =====================================================================
    function cardHeader(title, badgeHtml = '', subtitle = '') {
        return UI().cardHeader(title, badgeHtml, subtitle);
    }

    return {
        dashboard,
        project,
        resultsGlobal,
        outputsGlobal,
        reportsGlobal,
        auditGlobal,
        showNewProjectForm,
        hideNewProjectForm,
        loadProjectList,
        showUploadForm,
        hideUploadForm,
        runDataQuality,
        generateAnalysisPlan,
        reviewAnalysis,
        approveAnalysis,
        executeAnalysis,
        closeModal,
        showProvenance,
        validateOutput,
        renderOutput,
        approveOutput,
        downloadOutput,
        createReproPackage,
        verifyReproPackage,
        viewReproManifest,
        cardHeader,
    };
})();

if (typeof window !== 'undefined') window.STATIS_VIEWS = STATIS_VIEWS;
