import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const { requireRuntimeModule } = await import(pathToFileURL('C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.22227/skills/presentations/container_tools/runtime_helpers.mjs').href);

const root = 'C:/Users/Administrator/Documents/Codex/2026-07-13/gai/USV_CISF_SAC_729_3DOF_BODY_FRAME';
const visuals = path.join(root, 'visuals');
await fs.mkdir(visuals, {recursive:true});
const sharp = await requireRuntimeModule('sharp');
const W=1600,H=900;
const navy='#0B1F33', blue='#164A70', cyan='#18A6B8', pale='#F2F5F7', ink='#142735', muted='#5B6B78', white='#FFFFFF';
const slides=[
 ['海事规则内化流程与关键模块','面向 3DOF 无人艇安全追踪的 COLREGs 规则学习框架','SAC + COLREGs 动作过滤器 + 双批次规则模仿'],
 ['研究背景与任务定义','有界海洋环境中的单 USV 安全追踪','到达目标邻域，同时规避静态障碍、动态船舶、边界与危险会遇'],
 ['规则内化要解决的问题','安全层修正动作，不等于策略已经学会规则','需要把规则教师信号送入 Actor，并在测试时区分辅助安全与策略内化'],
 ['海事规则内化总流程','观测 → 规则评估 → 会遇分类 → 规则动作 → 平滑执行 → 教师记录 → Actor 双批次更新 → 过滤器关闭评估','规则池只服务 Actor；主池保持 SAC 训练语义'],
 ['规则过滤器：构成方式','COLREGsActionSetFilter','输入 world / agent / raw_action；输出动作链、模式、原因、会遇类型与 DCPA/TCPA'],
 ['规则过滤器：具体作用','按会遇类型与角色投影到规则动作集合','对遇、交叉让路、追越让路、直航与未定义风险分别处理'],
 ['规则状态器：滞回与义务锁存','RISK_DETECTED · GIVE_WAY · STAND_ON','进入阈值锁存义务，退出阈值连续确认，最大持有步数提供保护'],
 ['规则教师数据：动作链与记录字段','raw → constrained → rule → smoothed → applied','状态历史、动作层级、会遇编码、过滤模式、DCPA/TCPA 与教师置信度'],
 ['双批次经验池：构成方式','MainReplayBuffer + RuleReplayBuffer','主池服务 Critic / SAC Actor / Alpha / PER；规则池提供第二个 Actor 批次'],
 ['模仿学习损失：构成与作用','L_actor = L_SAC(B_main) + w_rule · L_rule(B_rule)','纯规则动作监督 Actor；两项损失一次反向传播，规则批次不进入 Critic 目标'],
 ['训练时序与开关逻辑','收集 → 写入规则池 → 预热 → 双批次更新 → 独立保存与恢复','规则池可选；未达到 RULE_BUFFER_MIN_SIZE 时规则项为零'],
 ['评估：如何判断“内化”','规则跟随率 × 安全化解率 → 内化成功率','事件级审计；risk_event_count=0 时为 N/A，不解释为 0% 学会'],
 ['项目评估记录：过滤器开关对照','0911 泛化评估，100 个配对 episode','policy-only：成功率 56%，碰撞率 30%；assisted：成功率 78%，碰撞率 0%'],
 ['方法贡献与已知边界','模块化规则链路 + 事件级审计 + Actor-only 规则监督','缺少跨种子、独立消融与统计显著性结论；原文未明确说明'],
 ['结论与展望','规则判断、动作过滤、教师记录、双批次更新与过滤器关闭评估构成闭环','补充多种子、消融与置信区间，持续检验策略自身的规则内化'],
];
const imgPaths=[];
const trackPath='D:/USV/logs/SAC_3DOF_BODY_FRAME_ENTITY_ENCODER_0911/evaluation/CUSTOM_BODY_FRAME_GENERALIZATION_DEMO/20260913_182700/episode_002_seed_1023152478_assisted.png';
const trackData=`data:image/png;base64,${(await fs.readFile(trackPath)).toString('base64')}`;
for(let i=0;i<slides.length;i++){
  const [title,subtitle,detail]=slides[i];
  let overlay='';
  if(i===12){
    overlay=`<rect x="940" y="175" width="590" height="505" rx="6" fill="#FFFFFF" stroke="${cyan}" stroke-width="3"/><image href="${trackData}" x="950" y="185" width="570" height="485" preserveAspectRatio="xMidYMid meet" opacity="0.96"/>`;
  }
  const body = i===12 ? `<text x="88" y="300" font-size="34" fill="${ink}">配对评估（100 回合）</text><text x="88" y="370" font-size="30" fill="${blue}">policy-only  56% 成功率 · 30% 碰撞率</text><text x="88" y="425" font-size="30" fill="${cyan}">assisted     78% 成功率 · 0% 碰撞率</text><text x="88" y="500" font-size="24" fill="${muted}">平均最小净距：17.75 m → 24.44 m</text><text x="88" y="545" font-size="24" fill="${muted}">内化成功率均值：4.00% → 8.87%</text>` : `<text x="88" y="300" font-size="34" fill="${ink}">${subtitle}</text><text x="88" y="380" font-size="27" fill="${muted}">${detail}</text>`;
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><rect width="100%" height="100%" fill="${pale}"/><rect width="100%" height="118" fill="${navy}"/><rect x="0" y="118" width="100%" height="7" fill="${cyan}"/><text x="88" y="78" font-family="Arial,Microsoft YaHei" font-size="40" font-weight="700" fill="${white}">${title}</text><text x="88" y="760" font-family="Arial,Microsoft YaHei" font-size="22" fill="${muted}">USV_CISF_SAC_729_3DOF_BODY_FRAME · 源码与项目日志证据</text><text x="1510" y="760" text-anchor="end" font-family="Arial" font-size="22" fill="${muted}">${String(i+1).padStart(2,'0')} / ${slides.length}</text><path d="M88 700 H1510" stroke="${cyan}" stroke-width="3" opacity=".7"/>${body}${overlay}<g opacity=".22" stroke="${blue}" fill="none"><path d="M1150 690 C1240 600 1260 480 1180 400 S1110 240 1250 170" stroke-width="5"/><circle cx="1250" cy="170" r="16" fill="${cyan}"/></g></svg>`;
  const svgPath=path.join(visuals,`slide-${String(i+1).padStart(2,'0')}.svg`); const pngPath=path.join(visuals,`slide-${String(i+1).padStart(2,'0')}.png`);
  await fs.writeFile(svgPath,svg); await sharp(Buffer.from(svg)).png().toFile(pngPath); imgPaths.push(pngPath);
}
const pres=Presentation.create({slideSize:{width:W,height:H}}); const family='Arial';
for(let i=0;i<imgPaths.length;i++){const s=pres.slides.add(); s.background.fill=pale; const b=await fs.readFile(imgPaths[i]); s.images.add({blob:b,contentType:'image/png',position:{left:0,top:0,width:W,height:H},fit:'cover',alt:slides[i][0]}); s.speakerNotes.textFrame.setText('来源：项目源码、README、配置文件与 0911 评估 CSV。');}
const candidate=path.join(root,'ppt_build','candidate.pptx'); await (await PresentationFile.exportPptx(pres)).save(candidate); console.log(candidate);
