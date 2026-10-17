export const meta = {
  name: "active-perception-positioning-audit",
  description: "Audit research positioning of an arm-mounted RGBD Fisher-information active-view closed-loop mapping system vs phone 3D scanning and the NBV literature's discrete-view paradigm.",
  phases: ["survey", "synthesize", "adversarial", "recommend"]
};

const REPO = "/home/fuyx/lanzc/omnimap";

const SYS = `
SYSTEM UNDER AUDIT (repo ${REPO}; you may read README.md, docs/, info_flow/, omnimap/gaussian/renderer/nbv/, sim/ for ground truth):
- Base: OmniMap online mapping = 3DGS optical + voxel TSDF geometry + open-vocab semantics (YOLO-World + TAP).
- Added: FisherRF-style Fisher information gain as view-evaluation criterion.
- Hardware: UR5 arm, eye-in-hand RealSense RGB-D, ROS Noetic, continuous RGB-D video stream.
- Fisher: diagonal Fisher = grad^2 of render wrt gaussian xyz+opacity. History info = sum over keyframes. Current-view score = sum(cur_stat/(hist_stat+lambda)).
- Control: central-difference dF/dtheta, dF/dphi on a view hemisphere -> normalize sign(g)*log1p(|g|/N_gauss) -> tangential spherical angular rates -> servo converts to Cartesian TwistStamped via LOCAL TF (tangential + radial PI holding reference sphere radius + look-at P for orientation).
- MAIN ENGINEERING CLAIM: decomposition into THREE independently-clocked loops. Tracking (keyframe-driven, few Hz, GPU) -> Planning (fixed 30 Hz, GPU) -> ServoAction (fixed 50 Hz, execution host, NO GPU/torch). Inter-loop message carries only POSE-INDEPENDENT quantities (theta_rate, phi_rate, reference_radius, reference_scene_center, model_version); the heavy frozen map state goes to a file-backed PlannerSnapshot referenced by URI, never over ROS.
- MOTIVATION (real failures on hardware): the serial single-process loop that worked in simulation FAILED on the real arm with (1) stale commands at single-digit Hz because OMNI.track is slow -> jerky motion, overshoot, cannot track; (2) "singularity lockup"/local-plateau oscillation: camera stalls near hemisphere poles or Fisher plateaus, commands chatter with ~zero net displacement. Fixes: raise control rate, gradient normalization by gaussian count, active stop below speed threshold, three independent velocity clamps + slew-rate limits + deadbands, radial P->PI to kill steady-state radius error.
- Also present: point-cloud-driven single-process sim closed loop (sim/), original offline Replica/ScanNet mapping + offline TSDF mesh.
- STATUS: a working engineering system. NO paper-grade experiments, baselines, or quantitative metrics exist yet.

THE USER'S TWO DOUBTS (the crux):
D1. Consumer phone apps reconstruct a mug by just waving a phone -- no arm, no depth sensor. So why does this hardware/complexity matter? Are they "just NeRF"?
D2. Frontier active-view-planning papers do NOT use continuous video; they use ONE photo per selected viewpoint (discrete capture at a pose). A continuous-stream + velocity-control formulation looks off-paradigm.
User's conclusion: "I can't see the justification for this work." Give an honest, evidence-grounded evaluation plus concrete recommendations.
`;

const NOTE = `
RULES: Use WebSearch/WebFetch aggressively; today is September 2026, so explicitly search 2024/2025/2026 work -- do NOT rely on memory alone. Cite title + venue + year + arXiv id. Explicitly flag anything you could NOT verify as UNVERIFIED. Be dense, skeptical, concrete. No preamble, no padding. Answer in English (the final synthesis will be translated).
`;

phase("survey");
const survey = await parallel([

  () => agent(`${SYS}\n${NOTE}\nTASK A -- Consumer phone 3D scanning: what is ACTUALLY under the hood?
Investigate real pipelines of current object-scanning products AND their open counterparts: Apple Object Capture / RealityKit, Polycam, Luma AI, Scaniverse (Niantic), KIRI Engine, RealityScan (Epic), Matterport, plus open pipelines (COLMAP/GLOMAP -> 3DGS/Nerfstudio, Meshroom).
Answer precisely:
1. For each: photogrammetry MVS, NeRF, 3DGS, feed-forward transformer reconstruction, or hybrid? Which use phone LiDAR/ToF or ARKit/ARCore depth+VIO pose priors, and which are pure RGB?
2. Is capture really "just wave the phone"? Document the guidance/constraints imposed on the user: orbit coverage requirements, on-screen coverage guidance, overlap requirements, frame selection/keyframe gating from a video stream, turntable modes, typical image counts, capture time.
3. Compute model: on-device vs cloud; reconstruction wall-clock; real-time incremental vs batch offline (SfM first, then optimize)?
4. Where do they FAIL, with evidence: textureless/specular/transparent, thin structures, metric scale, unobserved regions (mug bottom, concavities), lighting change, drift on large scenes, no semantics, no completeness guarantee, no closed-loop guarantee coverage is sufficient.
5. Crucially: do they solve the SAME PROBLEM as an autonomous robot deciding where to look next with no human in the loop? Characterize exactly what work the HUMAN operator performs in the phone pipeline -- that human is the hidden component the user is comparing against.
Return: comparison table + crisp verdict on "is phone scanning just NeRF?" + list of structural limitations of the phone pipeline that a robot-mounted online active system does not share.`),

  () => agent(`${SYS}\n${NOTE}\nTASK B -- The 2024-2026 pose-free / feed-forward / streaming reconstruction frontier and its impact on the user's premises.
Survey and characterize: DUSt3R, MASt3R and MASt3R-SLAM, VGGT (and any VGGT-SLAM/VGGT-Long), Fast3R, CUT3R, Spann3R, SLAM3R, StreamVGGT, Pi3, MoGe/MoGe-2, metric monocular depth (Depth Anything v2/v3, UniDepth, Metric3D), feed-forward GS (Splatt3R, NoPoSplat, LongSplat), and any 2026 successors. Also online/incremental 3DGS SLAM: MonoGS, SplaTAM, Photo-SLAM, Gaussian-SLAM, RTG-SLAM, and 2025/2026 successors.
Answer:
1. Which of these actually consume a CONTINUOUS VIDEO STREAM incrementally vs a batch of unordered images? Which are causal/online vs requiring the whole sequence? Give concrete evidence.
2. Does monocular RGB now suffice for metric-scale object-level reconstruction, or is RGB-D / known scale still needed? What is the honest state of metric scale in Sept 2026?
3. Do any of these support ACTIVE control -- i.e. does the reconstruction expose an uncertainty signal usable for closed-loop next-view decisions? Which ones publish uncertainty at all?
4. Directly assess: does this literature make an RGB-D + robot-arm + Fisher-gradient continuous-control system OBSOLETE, or does it just change which parts are novel? Be specific about what it obsoletes and what it does not.
5. Is a "continuous video stream" input actually unusual in reconstruction/SLAM research, or is it the NORM there and only unusual in the NBV/active-vision subfield? This distinction matters enormously for the user's doubt D2.`),

  () => agent(`${SYS}\n${NOTE}\nTASK C -- Active view planning / NBV / active perception: verify the user's claim that the field is exclusively "one photo per viewpoint".
Survey 2023-2026: FisherRF (ICLR/ECCV? verify), ActiveNeRF, ActiveGS, NARUTO, GauSS-N/uncertainty-aware GS NBV, ActiveSplat, ActiveGAMER, FisherRF successors, active mapping / informative path planning (IPP), next-best-view for object reconstruction, active SLAM surveys, and any 2026 work. ALSO deliberately search for the exceptions: continuous-time / receding-horizon / gradient-field / potential-field / differentiable-uncertainty view planning; "informative path planning" with continuous trajectories; "perception-aware" MPC/trajectory optimization; active vision on real manipulators; visual servoing driven by reconstruction uncertainty; drone/UAV active reconstruction with continuous flight.
Answer:
1. Is the "one photo for one viewpoint" (discrete candidate-set evaluate-then-teleport) paradigm actually dominant? Quantify roughly how many of the surveyed works do this.
2. WHY is it dominant? Separate genuine methodological reasons from convenience: candidate-set argmax is easy to evaluate/ablate; simulation teleports are free; stop-and-shoot avoids motion blur and pose error; batch training between views; reviewers can compare on fixed benchmarks.
3. What are the DOCUMENTED costs of stop-and-shoot in the literature, and does anyone criticize it? Look for: dead time, unusable inter-view frames, robot deceleration/acceleration, total scan wall-clock, non-holonomic/kinematic infeasibility of teleporting, and myopia of one-step-greedy argmax.
4. Find and list every work you can that DOES use continuous motion / continuous-time information gain / gradient-following on an information field -- especially on real hardware. Include drone work (e.g. FUEL, exploration with continuous trajectories), perception-aware planning, and any Fisher-information-field continuous control.
5. Verdict: is the user's system "off-paradigm and therefore wrong", or "off-paradigm and therefore potentially a contribution"? Argue both sides with citations, then commit to an answer.`),

  () => agent(`${SYS}\n${NOTE}\nTASK D -- What is actually novel here, and what would a top-venue paper need? Read the repo first.
Read ${REPO}/README.md, ${REPO}/docs/*.md, ${REPO}/omnimap/gaussian/renderer/nbv/motion_policy.py, ${REPO}/omnimap/gaussian/renderer/nbv/legacy_fisher.py, ${REPO}/info_flow/*.py, ${REPO}/sim/sim_fisher_closed_loop.py. Use grep/sed via Bash. Do NOT modify any file.
Then answer:
1. Inventory of what is genuinely implemented vs claimed. Be precise about the Fisher computation, the gradient normalization, the three-loop decomposition, the snapshot mechanism, the safety-degradation gates, and the servo control law.
2. Which of these are (a) novel research contributions, (b) solid systems engineering worth a systems-venue paper, (c) standard practice repackaged? Judge honestly and separately.
3. Search the literature (WebSearch) specifically for prior art on: asynchronous/decoupled mapping-vs-planning-vs-control rate decomposition in active perception; "snapshot"/frozen-map-state mechanisms for planning against a slow map; information-gradient following as continuous control; radial-PI + tangential-field control on a view sphere. Is the three-loop decomposition novel, or is it standard robotics practice (e.g. classic 3-tier/sense-plan-act, MPC hierarchies) applied to a new domain?
4. What is the single strongest publishable claim available from this system AS BUILT? What is the second? Rank by defensibility.
5. What experiments/metrics/baselines would be required to support each claim? Be brutally concrete: what to measure, what to compare against, what ablation isolates the contribution, and which claims are UNSUPPORTABLE without new hardware or new datasets.
6. Which venues fit (ICRA/IROS/RA-L/CoRL/T-RO vs CVPR/ICCV/NeurIPS)? Explain the fit-per-claim, since the vision venues and robotics venues want different things here.`),

]);

const [A, B, C, D] = survey;

const EVID = `
=== EVIDENCE FROM PARALLEL SURVEY ===
--- TASK A: phone/consumer 3D scanning pipelines ---
${A}
--- TASK B: pose-free / feed-forward / streaming reconstruction frontier ---
${B}
--- TASK C: NBV / active perception paradigm audit ---
${C}
--- TASK D: repo-grounded novelty and paper-readiness audit ---
${D}
=== END EVIDENCE ===
`;

phase("synthesize");
const synth = await agent(`${SYS}\n${NOTE}\n${EVID}
TASK -- Write the honest positioning verdict. You are a senior researcher who has read all the above evidence. The user is a graduate student having a crisis of confidence about their own work. They deserve neither flattery nor demolition -- they deserve accuracy.
Produce a structured English report:
1. ANSWER D1 DIRECTLY: how phone scanning works, whether it is "just NeRF", and -- decisively -- whether it solves the same problem. Name the hidden human-in-the-loop component explicitly. State plainly which parts of the user's premise are correct (some are) and which conflate different problems.
2. ANSWER D2 DIRECTLY: is the field really one-photo-per-viewpoint? If mostly yes, is that a law of nature or a convenience artifact? Say explicitly whether the user's continuous-stream formulation is a liability, a neutral choice, or a latent contribution -- and under exactly what framing it becomes a contribution rather than a deviation.
3. HONEST WEAKNESS AUDIT of the work as it stands. This must be genuinely critical -- list the things a reviewer would attack: greedy one-step Fisher gradient with no horizon, hemisphere/sphere parameterization restricting the workspace and creating the pole singularity, an object-centric reference sphere assumption, no baselines, no metrics, no ablations, unclear whether continuous motion actually beats stop-and-shoot on any measurable axis, whether the three-loop split is engineering rather than science, motion blur and pose error under continuous motion, whether 3DGS-based Fisher is even the right uncertainty, and the fact that arm-mounted eye-in-hand on a UR5 is a very constrained workspace.
4. WHAT IS ACTUALLY DEFENSIBLE. Be specific and ranked. Distinguish "novel research claim" from "strong systems paper" from "good engineering with no paper in it".
5. THE REFRAMING that makes this work make sense -- or state clearly if there is none. If the honest answer is "this is a systems/engineering contribution, not a vision-novelty contribution", say that plainly and say what venue rewards it.
Do not hedge into mush. Where the evidence is thin, say so.`);

phase("adversarial");
const adver = await parallel([

  () => agent(`${SYS}\n${NOTE}\n${EVID}
=== DRAFT VERDICT TO ATTACK ===
${synth}
=== END DRAFT ===
TASK -- You are Reviewer 2 at ICRA/CoRL, hostile and well-read. Write the strongest possible REJECTION of both (a) the user's system and (b) the draft verdict above where it is too generous.
Cover: is continuous-motion active reconstruction a solved/uninteresting problem given drone exploration literature (FUEL, perception-aware MPC) already does continuous information-driven flight? Is the three-loop decomposition just textbook asynchronous ROS architecture dressed up? Is Fisher-on-3DGS the wrong uncertainty measure? Is the whole premise dead because feed-forward pose-free reconstruction from a casually waved phone will absorb this application? Is the UR5 tabletop setting too toy to matter? Would any measured improvement be attributable to the contribution rather than to tuning?
Then, separately, list which of your own attacks are FATAL vs which are ADDRESSABLE with the right experiment or reframing. Be precise about what experiment would defuse each addressable attack.`),

  () => agent(`${SYS}\n${NOTE}\n${EVID}
=== DRAFT VERDICT TO PRESSURE-TEST ===
${synth}
=== END DRAFT ===
TASK -- You are the strongest possible ADVOCATE, but an honest one. Build the best case that this work is not only justified but points at something the field is actually missing.
Specifically develop, with citations: (1) the "wall-clock time to a given reconstruction quality" axis -- is total scan time an under-measured metric in the NBV literature, and would continuous motion win on it? Find any evidence for or against. (2) The "every frame is information" argument: stop-and-shoot throws away all inter-view frames; with online 3DGS SLAM those frames are free supervision. Is there literature quantifying this? (3) The embodiment argument: a robot cannot teleport, so candidate-set NBV silently ignores path cost and kinematic feasibility -- who has pointed this out? (4) The verification/closure argument: a robot can guarantee coverage; a human waving a phone cannot -- does industrial/metrology inspection literature support this framing? (5) Where does an active RGB-D arm system remain strictly necessary in Sept 2026 (specularity, transparency, metrology tolerance, dark/textureless, repeatability, no-human-available settings)?
Also: identify the ONE experiment that would most change a skeptic's mind, and describe it precisely enough to implement.`),

]);

const [attack, defense] = adver;

phase("recommend");
const rec = await agent(`${SYS}\n${NOTE}\n${EVID}
=== SYNTHESIS ===
${synth}
=== HOSTILE REVIEW ===
${attack}
=== HONEST ADVOCACY ===
${defense}
=== END INPUTS ===
TASK -- Produce the FINAL deliverable for the user. Write it in CHINESE (the user writes Chinese), technical register, no flattery, no filler. Resolve the disagreements between the hostile review and the advocacy explicitly -- where they conflict, adjudicate and say which side wins and why.

Required structure:

## 一、手机扫描到底怎么做的（回答疑问 1）
Concrete pipelines, who uses what, whether it is NeRF/3DGS/MVS/feed-forward, role of ARKit VIO + LiDAR, batch-vs-online, and the decisive point: what the human operator is silently contributing. State plainly which part of the user's premise is right.

## 二、"一视角一张图" 是不是铁律（回答疑问 2）
Whether it dominates, WHY (separate real reasons from convenience), what it costs, who deviates. Then the verdict on whether continuous stream is a liability or an opening.

## 三、这个工作的真实处境
An honest ledger: 站得住的 / 站不住的 / 尚未验证的. Include the hostile review's FATAL attacks verbatim in substance -- do not soften them.

## 四、重新定位：三个可选叙事
Give exactly three candidate framings, each with: 一句话 claim, 为什么成立, 需要的实验, 目标会议, 风险. Rank them and recommend one. At least one framing must be honest about being a systems/robotics contribution rather than a vision-novelty one.

## 五、最该做的实验（按性价比排序）
A concrete prioritized list. For each: what to measure, against what baseline, what result would support the claim, roughly what it costs to run on the existing UR5 + sim setup. The top item must be the single skeptic-flipping experiment. Be explicit that the stop-and-shoot baseline must be implemented on the SAME system for the comparison to mean anything, and note that the existing sim/ closed loop plus the offline Replica/ScanNet path can carry much of this without new hardware.

## 六、如果结果是负面的怎么办
Brutally honest contingency: what if continuous motion does NOT beat stop-and-shoot? What is the fallback narrative and is it still publishable? Say what would make this work unpublishable and what the exit options are (workshop paper, RA-L systems paper, thesis chapter, open-source release).

## 七、一句话结论
The single most useful sentence.

Rules: cite paper titles + years inline where they carry weight. Mark UNVERIFIED claims. Do not invent citations -- if the survey did not verify a paper, say so. Total length: thorough but every sentence must carry information.`);

rec;
