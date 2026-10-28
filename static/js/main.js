/**
 * Frontend Controller for the Classical NLP Smart Study Assistant
 * Coordinates:
 *  - Preset Chapter Selection, Custom Text Paste, and File Uploads (.txt/.md/.pdf)
 *  - Interactive Annotated Sentence Reader with HMM-Viterbi inline entity tags
 *  - Executive Summary (TextRank), Entity/Flashcard cards (HMM-Viterbi), and Q&A (BM25 + Naive Bayes + Lesk)
 *  - Live Numerical & Mathematical Inspector Drawer
 */

document.addEventListener("DOMContentLoaded", () => {
  const chapters = window.INITIAL_CHAPTERS || [];
  let activeText = "";
  let activeTitle = "";
  let currentPresetQuestions = [];

  // DOM Elements
  const chapterSelect = document.getElementById("chapterSelect");
  const fileUpload = document.getElementById("fileUpload");
  const uploadStatus = document.getElementById("uploadStatus");
  const summaryKInput = document.getElementById("summaryK");
  const dampingInput = document.getElementById("dampingFactor");
  const bm25K1Input = document.getElementById("bm25K1");
  const bm25BInput = document.getElementById("bm25B");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const docTitleHeading = document.getElementById("docTitleHeading");
  const toggleEditBtn = document.getElementById("toggleEditBtn");
  const rawEditorContainer = document.getElementById("rawEditorContainer");
  const customDocTitle = document.getElementById("customDocTitle");
  const rawTextInput = document.getElementById("rawTextInput");
  const applyCustomTextBtn = document.getElementById("applyCustomTextBtn");
  const cancelEditBtn = document.getElementById("cancelEditBtn");

  const annotatedReader = document.getElementById("annotatedReader");
  const summaryList = document.getElementById("summaryList");
  const entityGroupsContainer = document.getElementById("entityGroupsContainer");
  const flashcardsGrid = document.getElementById("flashcardsGrid");

  const questionChips = document.getElementById("questionChips");
  const questionInput = document.getElementById("questionInput");
  const askBtn = document.getElementById("askBtn");
  const qaResultContainer = document.getElementById("qaResultContainer");
  const qaExpectedType = document.getElementById("qaExpectedType");
  const qaExactAnswer = document.getElementById("qaExactAnswer");
  const qaBm25TopScore = document.getElementById("qaBm25TopScore");
  const qaSupportingSentence = document.getElementById("qaSupportingSentence");
  const qaTopPassagesList = document.getElementById("qaTopPassagesList");
  const qaLeskContainer = document.getElementById("qaLeskContainer");

  const toggleMathBtn = document.getElementById("toggleMathBtn");
  const toggleParamsBtn = document.getElementById("toggleParamsBtn");
  const paramsDrawer = document.getElementById("paramsDrawer");
  const closeMathBtn = document.getElementById("closeMathBtn");

  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = [
    document.getElementById("qaCard"),
    document.getElementById("summaryCard"),
    document.getElementById("flashcardsCard")
  ];
  const mathInspectorModal = document.getElementById("mathInspectorModal");
  const mathTextRankContent = document.getElementById("mathTextRankContent");
  const mathHmmContent = document.getElementById("mathHmmContent");
  const mathBm25Content = document.getElementById("mathBm25Content");
  const mathQaContent = document.getElementById("mathQaContent");

  // Initialize with the first preset chapter
  if (chapters.length > 0) {
    loadPresetChapter(chapters[0].id);
  }

  chapterSelect.addEventListener("change", (e) => {
    const val = e.target.value;
    if (val === "custom") {
      rawEditorContainer.classList.remove("hidden");
    } else {
      loadPresetChapter(val);
    }
  });

  toggleEditBtn.addEventListener("click", () => {
    rawEditorContainer.classList.toggle("hidden");
    rawTextInput.value = activeText;
    customDocTitle.value = activeTitle;
  });

  cancelEditBtn.addEventListener("click", () => {
    rawEditorContainer.classList.add("hidden");
  });

  applyCustomTextBtn.addEventListener("click", () => {
    const customText = rawTextInput.value.trim();
    const title = customDocTitle.value.trim() || "Custom Study Document";
    if (!customText) {
      alert("Please paste or enter some text to analyze.");
      return;
    }
    chapterSelect.value = "custom";
    currentPresetQuestions = [];
    rawEditorContainer.classList.add("hidden");
    runAnalyzePipeline(customText, title);
  });

  analyzeBtn.addEventListener("click", () => {
    if (!activeText.trim()) return;
    runAnalyzePipeline(activeText, activeTitle);
  });

  // File Upload (.txt, .md, .pdf)
  fileUpload.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadStatus.textContent = `Uploading & analyzing ${file.name}...`;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("summary_k", summaryKInput.value || 3);
    formData.append("damping_factor", dampingInput.value || 0.85);

    try {
      const resp = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        uploadStatus.textContent = `Error: ${data.error || "Upload failed"}`;
        return;
      }

      chapterSelect.value = "custom";
      activeText = data.raw_text || "";
      activeTitle = data.title || `Uploaded: ${file.name}`;
      rawTextInput.value = activeText;
      customDocTitle.value = activeTitle;
      currentPresetQuestions = [];
      uploadStatus.textContent = `Loaded: ${file.name} (${data.num_sentences} sentences)`;

      renderAnalysisOutput(data);
    } catch (err) {
      uploadStatus.textContent = `Upload error: ${err.message}`;
    }
  });

  // Ask Question Button & Enter Key
  askBtn.addEventListener("click", () => {
    const q = questionInput.value.trim();
    if (q) runQuestionPipeline(q);
  });

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const q = questionInput.value.trim();
      if (q) runQuestionPipeline(q);
    }
  });

  // Right-Pane 3-Tab Switcher (Q&A | Summary | Key Takeaways)
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      tabPanels.forEach((panel) => {
        if (!panel) return;
        if (panel.id === targetId) {
          panel.classList.remove("hidden");
        } else {
          panel.classList.add("hidden");
        }
      });
    });
  });

  // Toggle Hyperparameter Settings Drawer
  if (toggleParamsBtn && paramsDrawer) {
    toggleParamsBtn.addEventListener("click", () => {
      paramsDrawer.classList.toggle("hidden");
    });
  }

  // Toggle Math Drawer
  toggleMathBtn.addEventListener("click", () => {
    mathInspectorModal.classList.toggle("hidden");
    if (!mathInspectorModal.classList.contains("hidden")) {
      mathInspectorModal.scrollIntoView({ behavior: "smooth" });
    }
  });

  closeMathBtn.addEventListener("click", () => {
    mathInspectorModal.classList.add("hidden");
  });

  function loadPresetChapter(chapterId) {
    const found = chapters.find((c) => c.id === chapterId);
    if (!found) return;
    activeText = found.text;
    activeTitle = found.title;
    rawTextInput.value = activeText;
    customDocTitle.value = activeTitle;
    currentPresetQuestions = found.preset_questions || [];
    uploadStatus.textContent = `Preset loaded (${found.category})`;
    runAnalyzePipeline(activeText, activeTitle);
  }

  async function runAnalyzePipeline(text, title) {
    activeText = text;
    activeTitle = title;
    docTitleHeading.textContent = title;

    const payload = {
      text: text,
      title: title,
      summary_k: parseInt(summaryKInput.value || "3", 10),
      damping_factor: parseFloat(dampingInput.value || "0.85")
    };

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      renderAnalysisOutput(data);
    } catch (err) {
      console.error("Failed to analyze document:", err);
    }
  }

  function renderAnalysisOutput(data) {
    docTitleHeading.textContent = data.title || "Study Document";

    const summarySentences = (data.summary && data.summary.summary_sentences) || [];
    const summaryIndexSet = new Set(summarySentences.map((s) => s.sentence_index));

    // 1. Render Annotated Document Reader
    annotatedReader.innerHTML = "";
    (data.sentences || []).forEach((sObj) => {
      const div = document.createElement("div");
      div.className = "doc-sentence" + (summaryIndexSet.has(sObj.index) ? " is-summary" : "");
      div.id = `doc-sent-${sObj.index}`;
      div.innerHTML = `<span class="sent-idx-badge">[S${sObj.index + 1}]</span> ` +
        formatSentenceWithEntities(sObj.text, sObj.annotations);
      div.addEventListener("click", () => highlightSentenceInReader(sObj.index));
      annotatedReader.appendChild(div);
    });

    // 2. Render Executive Summary (Algorithm 1: TextRank)
    summaryList.innerHTML = "";
    summarySentences.forEach((item) => {
      const li = document.createElement("li");
      li.className = "summary-item";
      li.innerHTML = `
        <div class="summary-meta">
          <span>Sentence [S${item.sentence_index + 1}] (Rank #${item.rank})</span>
          <span>PageRank Centrality: ${item.pagerank_score}</span>
        </div>
        <div>${escapeHtml(item.text)}</div>
      `;
      li.addEventListener("click", () => highlightSentenceInReader(item.sentence_index));
      summaryList.appendChild(li);
    });

    // 3. Render HMM-Viterbi Entities & Flashcards (Algorithm 2)
    const entitiesByType = (data.information_extraction && data.information_extraction.entities_by_type) || {};
    renderEntityGroups(entitiesByType);

    const flashcards = (data.information_extraction && data.information_extraction.flashcards) || [];
    renderFlashcards(flashcards);

    // 4. Render Suggested Questions (Preset or Auto-generated for custom uploads)
    const questionsToUse =
      currentPresetQuestions.length > 0
        ? currentPresetQuestions
        : data.suggested_questions || [];
    renderQuestionChips(questionsToUse);

    // Automatically run the first question so the Q&A panel and math inspector are populated immediately
    if (questionsToUse.length > 0) {
      questionInput.value = questionsToUse[0];
      runQuestionPipeline(questionsToUse[0], false);
    } else {
      qaResultContainer.classList.add("hidden");
    }

    // 5. Render Math Inspector for Algorithm 1 (TextRank) and Algorithm 2 (HMM-Viterbi)
    renderTextRankMath(data.summary ? data.summary.math_details : null);
    renderHmmMath(data.information_extraction ? data.information_extraction.math_details : null);
  }

  function formatSentenceWithEntities(rawText, annotations) {
    if (!annotations || !annotations.tokens || !annotations.tags) {
      return escapeHtml(rawText);
    }
    const tokens = annotations.tokens;
    const tags = annotations.tags;
    let htmlParts = [];
    let i = 0;
    while (i < tokens.length) {
      const tag = tags[i];
      if (tag && tag !== "OTHER") {
        let spanTokens = [tokens[i]];
        let j = i + 1;
        while (j < tokens.length && tags[j] === tag) {
          spanTokens.push(tokens[j]);
          j++;
        }
        const phrase = escapeHtml(spanTokens.join(" "));
        htmlParts.push(
          `<span class="inline-entity tag-${tag}">${phrase}<sub>${tag}</sub></span>`
        );
        i = j;
      } else {
        htmlParts.push(escapeHtml(tokens[i]));
        i++;
      }
    }
    // Clean spacing before punctuation
    return htmlParts.join(" ").replace(/\s+([.,!?;:])/g, "$1");
  }

  function renderEntityGroups(entitiesByType) {
    entityGroupsContainer.innerHTML = "";
    const labels = [
      { key: "PERSON", title: "People & Figures (PERSON)" },
      { key: "LOCATION", title: "Locations & Labs (LOCATION)" },
      { key: "DATE", title: "Dates & Metrics (DATE)" },
      { key: "CONCEPT", title: "Core Technical Concepts (CONCEPT)" }
    ];

    labels.forEach(({ key, title }) => {
      const list = entitiesByType[key] || [];
      const box = document.createElement("div");
      box.className = "entity-group-box";
      let chipsHtml = "";
      if (list.length === 0) {
        chipsHtml = `<span class="muted-note">None detected</span>`;
      } else {
        chipsHtml = list
          .slice(0, 8)
          .map(
            (item) =>
              `<span class="entity-pill tag-${key}" data-sidx="${item.sentence_index}">${escapeHtml(item.entity)}</span>`
          )
          .join("");
      }
      box.innerHTML = `
        <div class="entity-group-title">${title}</div>
        <div class="entity-chip-list">${chipsHtml}</div>
      `;
      box.querySelectorAll(".entity-pill").forEach((pill) => {
        pill.addEventListener("click", () => {
          const sidx = parseInt(pill.getAttribute("data-sidx"), 10);
          highlightSentenceInReader(sidx);
        });
      });
      entityGroupsContainer.appendChild(box);
    });
  }

  function renderFlashcards(flashcards) {
    flashcardsGrid.innerHTML = "";
    flashcards.slice(0, 6).forEach((card) => {
      const div = document.createElement("div");
      div.className = "flashcard";
      div.innerHTML = `
        <div class="flashcard-cat">${escapeHtml(card.category)}</div>
        <div class="flashcard-front">${escapeHtml(card.front)}</div>
        <div class="flashcard-back hidden">
          <strong>Answer Context [S${card.sentence_index + 1}]:</strong> ${escapeHtml(card.back)}
        </div>
      `;
      div.addEventListener("click", () => {
        const backEl = div.querySelector(".flashcard-back");
        backEl.classList.toggle("hidden");
        highlightSentenceInReader(card.sentence_index);
      });
      flashcardsGrid.appendChild(div);
    });
  }

  function renderQuestionChips(questions) {
    questionChips.innerHTML = "";
    questions.forEach((q) => {
      const btn = document.createElement("button");
      btn.className = "q-chip";
      btn.textContent = q;
      btn.addEventListener("click", () => {
        questionInput.value = q;
        runQuestionPipeline(q, true);
      });
      questionChips.appendChild(btn);
    });
  }

  async function runQuestionPipeline(question, focusReader = true) {
    const payload = {
      question: question,
      text: activeText,
      k1: parseFloat(bm25K1Input.value || "1.5"),
      b: parseFloat(bm25BInput.value || "0.75")
    };

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      renderQaResult(data, focusReader);
    } catch (err) {
      console.error("Failed to run Q&A pipeline:", err);
    }
  }

  function renderQaResult(data, focusReader) {
    qaResultContainer.classList.remove("hidden");
    const bm25 = data.retrieval_bm25 || {};
    const qa = data.qa_lesk || {};

    qaExpectedType.textContent = qa.expected_answer_type || "CONCEPT";
    qaExpectedType.className = `eat-badge tag-${qa.expected_answer_type || "CONCEPT"}`;
    qaExactAnswer.textContent = qa.exact_answer || "No exact span found.";

    const topPassage = (bm25.top_passages && bm25.top_passages[0]) || null;
    const topScore = topPassage ? topPassage.bm25_score : 0.0;
    qaBm25TopScore.textContent = `BM25 Score: ${topScore}`;

    const sentIdx = qa.sentence_index !== undefined ? qa.sentence_index : -1;
    qaSupportingSentence.innerHTML =
      sentIdx >= 0
        ? `<strong>[S${sentIdx + 1}]</strong> ${escapeHtml(qa.supporting_sentence)}`
        : escapeHtml(qa.supporting_sentence || "");

    qaSupportingSentence.onclick = () => {
      if (sentIdx >= 0) highlightSentenceInReader(sentIdx);
    };

    if (focusReader && sentIdx >= 0) {
      highlightSentenceInReader(sentIdx);
    }

    // Top 3 BM25 Candidate Passages summary
    qaTopPassagesList.innerHTML = (bm25.top_passages || [])
      .map(
        (p, idx) =>
          `<div>#${idx + 1} [S${p.sentence_index + 1}] (BM25 = <strong>${p.bm25_score}</strong>, len=${p.doc_length}): ${escapeHtml(p.text.slice(0, 95))}...</div>`
      )
      .join("");

    // Lesk Word Sense Disambiguation Cards
    const leskList = qa.lesk_disambiguations || [];
    if (leskList.length === 0) {
      qaLeskContainer.innerHTML = `<div class="muted-note">No polysemous words required disambiguation in this query.</div>`;
    } else {
      qaLeskContainer.innerHTML = leskList
        .map(
          (item) => `
          <div class="lesk-card">
            <div class="lesk-word-header">
              <span>Term: "${escapeHtml(item.word)}" &rarr; ${escapeHtml(item.selected_sense)}</span>
              <span>Overlap Score: ${item.overlap_score} (${escapeHtml(item.domain)})</span>
            </div>
            <div><strong>Disambiguated Definition:</strong> ${escapeHtml(item.definition)}</div>
            <div style="font-size:0.74rem; color:#94a3b8; margin-top:0.2rem;">
              Context-Gloss Stem Intersection: <code>{${(item.overlapping_stems || []).join(", ")}}</code>
            </div>
          </div>
        `
        )
        .join("");
    }

    // Update Math Inspector for Algorithm 3 (BM25) & Algorithm 4 (Naive Bayes + Lesk)
    renderBm25Math(bm25.math_details);
    renderQaMath(qa.math_details);
  }

  function highlightSentenceInReader(sentenceIndex) {
    document.querySelectorAll(".doc-sentence").forEach((el) => {
      el.classList.remove("active-focus");
    });
    const target = document.getElementById(`doc-sent-${sentenceIndex}`);
    if (target) {
      target.classList.add("active-focus");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  // ==================== MATHEMATICAL INSPECTOR RENDERERS ====================

  function renderTextRankMath(math) {
    if (!math) return;
    const simPreview = math.similarity_matrix_preview || [];
    let matrixHtml = `<table class="math-table"><thead><tr><th>Sent</th>`;
    simPreview.forEach((_, idx) => {
      matrixHtml += `<th>S${idx + 1}</th>`;
    });
    matrixHtml += `</tr></thead><tbody>`;
    simPreview.forEach((row, rIdx) => {
      matrixHtml += `<tr><th>S${rIdx + 1}</th>`;
      row.forEach((val) => {
        matrixHtml += `<td>${val}</td>`;
      });
      matrixHtml += `</tr>`;
    });
    matrixHtml += `</tbody></table>`;

    mathTextRankContent.innerHTML = `
      <div style="font-size:0.8rem; margin-bottom:0.4rem;">
        <strong>Sentences ($N$):</strong> ${math.num_sentences} |
        <strong>Unique Stems ($|V|$):</strong> ${math.vocabulary_size} |
        <strong>Converged in:</strong> ${math.iterations_to_converge} iterations ($L_1$ residual = ${math.final_l1_residual})
      </div>
      <div style="font-size:0.78rem; color:#94a3b8;">Cosine Similarity Adjacency Matrix $W_{i,j}$ (First ${math.matrix_preview_size} sentences):</div>
      ${matrixHtml}
      <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;">
        Stationary PageRank Eigenvector $p^*$: <code>[${(math.pagerank_vector || []).join(", ")}]</code>
      </div>
    `;
  }

  function renderHmmMath(math) {
    if (!math) return;
    const states = math.hidden_states || [];
    const trans = math.transition_matrix || [];

    let transHtml = `<table class="math-table"><thead><tr><th>From \\ To</th>`;
    states.forEach((s) => (transHtml += `<th>${s}</th>`));
    transHtml += `</tr></thead><tbody>`;
    trans.forEach((row, i) => {
      transHtml += `<tr><th>${states[i]}</th>`;
      row.forEach((val) => (transHtml += `<td>${val}</td>`));
      transHtml += `</tr>`;
    });
    transHtml += `</tbody></table>`;

    let trellisHtml = "";
    const sample = math.sample_viterbi_trellis;
    if (sample && sample.tokens) {
      trellisHtml += `<div style="font-size:0.78rem; color:#94a3b8; margin-top:0.6rem;">
        Sample Viterbi Log-Probability Trellis $V[k, t]$ on Sentence [S${sample.sentence_index + 1}]:
      </div>`;
      trellisHtml += `<table class="math-table"><thead><tr><th>State</th>`;
      sample.tokens.forEach((tok, idx) => {
        trellisHtml += `<th>${escapeHtml(tok)}<br/><span style="color:#38bdf8">${sample.decoded_tags[idx]}</span></th>`;
      });
      trellisHtml += `</tr></thead><tbody>`;
      sample.states.forEach((st, sIdx) => {
        trellisHtml += `<tr><th>${st}</th>`;
        sample.trellis_log_probs[sIdx].forEach((val) => {
          trellisHtml += `<td>${val}</td>`;
        });
        trellisHtml += `</tr>`;
      });
      trellisHtml += `</tbody></table>`;
    }

    mathHmmContent.innerHTML = `
      <div style="font-size:0.78rem; color:#94a3b8;">HMM Transition Probability Matrix $A_{i,j} = P(y_t = j \\mid y_{t-1} = i)$:</div>
      ${transHtml}
      ${trellisHtml}
    `;
  }

  function renderBm25Math(math) {
    if (!math) return;
    const idfTable = math.query_term_idf_table || [];
    let idfHtml = `<table class="math-table"><thead><tr><th>Stemmed Query Term $q$</th><th>Doc Freq $df(q)$</th><th>BM25 $IDF(q)$</th></tr></thead><tbody>`;
    idfTable.forEach((row) => {
      idfHtml += `<tr><td>${escapeHtml(row.stemmed_term)}</td><td>${row.doc_frequency}</td><td>${row.idf}</td></tr>`;
    });
    idfHtml += `</tbody></table>`;

    let topBreakHtml = "";
    if (math.top_passage_breakdown) {
      const tb = math.top_passage_breakdown;
      topBreakHtml = `<div style="font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;">
        Top Passage [S${tb.sentence_index + 1}] Length Normalization Factor $(1 - b + b \\cdot |D|/avgdl) = ${tb.length_norm_factor}$ (Total BM25 = <strong>${tb.bm25_score}</strong>)
      </div>`;
    }

    mathBm25Content.innerHTML = `
      <div style="font-size:0.8rem; margin-bottom:0.4rem;">
        <strong>Passages ($N$):</strong> ${math.total_passages_N} |
        <strong>Avg Length ($avgdl$):</strong> ${math.average_passage_length_avgdl} tokens |
        <strong>Params:</strong> $k_1 = ${math.k1}, b = ${math.b}$
      </div>
      ${idfHtml}
      ${topBreakHtml}
    `;
  }

  function renderQaMath(math) {
    if (!math || !math.naive_bayes) return;
    const nb = math.naive_bayes;
    const posteriors = nb.posterior_probabilities || {};
    const logProbs = nb.log_probabilities || {};

    let nbHtml = `<table class="math-table"><thead><tr><th>Answer Class $c$</th><th>Log-Score $\\log P(c, Q)$</th><th>Posterior $P(c \\mid Q)$</th></tr></thead><tbody>`;
    Object.keys(posteriors).forEach((cls) => {
      nbHtml += `<tr><th>${cls}</th><td>${logProbs[cls]}</td><td><strong>${posteriors[cls]}</strong></td></tr>`;
    });
    nbHtml += `</tbody></table>`;

    const spans = math.candidate_spans_scored || [];
    let spansHtml = "";
    if (spans.length > 0) {
      spansHtml = `<div style="font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;">Top Candidate Answer Spans (Type Alignment + Keyword Proximity Score):</div>`;
      spansHtml += `<table class="math-table"><thead><tr><th>Candidate Span</th><th>HMM Tag</th><th>Proximity</th><th>Final Score</th></tr></thead><tbody>`;
      spans.slice(0, 4).forEach((sp) => {
        spansHtml += `<tr><td>${escapeHtml(sp.span_text)}</td><td>${sp.hmm_tag}</td><td>${sp.proximity_score}</td><td><strong>${sp.final_score}</strong></td></tr>`;
      });
      spansHtml += `</tbody></table>`;
    }

    mathQaContent.innerHTML = `
      <div style="font-size:0.78rem; color:#94a3b8;">Multinomial Naive Bayes Question Classification:</div>
      ${nbHtml}
      ${spansHtml}
    `;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
});
