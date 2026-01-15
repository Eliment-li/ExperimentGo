import asyncio
import datetime as dt
import json
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import asyncssh
import streamlit as st
import yaml


# -----------------------------
# Utilities
# -----------------------------
def load_config(path: str = "experiments.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def _load_from_dir(dpath: str) -> list:
        out = []
        if not os.path.isdir(dpath):
            return out
        files = sorted([fn for fn in os.listdir(dpath) if fn.endswith((".yml", ".yaml"))])
        for fn in files:
            p = os.path.join(dpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as ef:
                    data = yaml.safe_load(ef) or {}
                if isinstance(data, list):
                    out.extend(data)
                elif isinstance(data, dict):
                    out.append(data)
            except Exception:
                # skip invalid files but continue loading others
                continue
        return out

    # If experiments key is a path string that points to a directory, load files from it
    exps_val = cfg.get("experiments")
    if isinstance(exps_val, str) and os.path.isdir(exps_val):
        cfg["experiments"] = _load_from_dir(exps_val)
    elif exps_val is None:
        # try default candidate directories next to the config file
        base_dir = os.path.dirname(path) or "."
        for cand in ("experiments", "experiments.d"):
            cand_path = os.path.join(base_dir, cand)
            if os.path.isdir(cand_path):
                cfg["experiments"] = _load_from_dir(cand_path)
                break

    return cfg


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "exp"


def now_compact() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def q(s: str) -> str:
    # safe shell quoting for values
    return shlex.quote(str(s))


def format_flag(name: str) -> str:
	return name.replace("_", "-")


def quote_path_for_shell(path: str) -> str:
	cleaned = str(path)
	if cleaned == "~":
		return '"$HOME"'
	if cleaned.startswith("~/"):
		tail = cleaned[2:]
		return f'"$HOME/{tail}"' if tail else '"$HOME"'
	return q(cleaned)


def trigger_rerun():
	if hasattr(st, "rerun"):
		st.rerun()
	else:
		st.experimental_rerun()


# -----------------------------
# Schema structures (lightweight)
# -----------------------------
@dataclass
class Host:
    name: str
    hostname: str
    port: int
    username: str
    workdir: str = "~"  # 每个 host 有自己的 workdir


@dataclass
class ArgSpec:
    name: str
    type: str  # int/float/str/bool/choice
    required: bool = False
    default: Any = None
    help: str = ""
    choices: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    bool_style: str = "presence"  # presence | negatable


@dataclass
class ExperimentSpec:
    name: str
    description: str
    module: str
    pre_cmds: List[str]
    args: List[ArgSpec]


def parse_specs(cfg: dict):
    hosts = []
    for h in cfg["ssh"]["hosts"]:
        hosts.append(Host(
            name=h["name"],
            hostname=h["hostname"],
            port=int(h.get("port", 22)),
            username=h["username"],
            workdir=h.get("workdir", "~"),
        ))

    runtime = cfg.get("runtime", {})
    conda_env = runtime.get("conda_env", "base")
    logs_dir = runtime.get("logs_dir", "~/exp-logs")

    exps = []
    for e in cfg.get("experiments", []):
        args = []
        for a in e.get("args", []):
            args.append(ArgSpec(
                name=a["name"],
                type=a["type"],
                required=bool(a.get("required", False)),
                default=a.get("default", None),
                help=a.get("help", ""),
                choices=a.get("choices"),
                min=a.get("min"),
                max=a.get("max"),
                bool_style=a.get("bool_style", "presence"),
            ))
        exps.append(ExperimentSpec(
            name=e["name"],
            description=e.get("description", ""),
            module=e["module"],
            pre_cmds=e.get("pre_cmds", []) or [],
            args=args,
        ))
    return hosts, conda_env, logs_dir, exps


# -----------------------------
# Command generation
# -----------------------------
def validate_and_build_args(spec: ExperimentSpec, values: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for a in spec.args:
        v = values.get(a.name, None)

        if v is None or (isinstance(v, str) and v == ""):
            if a.required:
                raise ValueError(f"参数 {a.name} 为必填")
            else:
                continue

        if a.type == "int":
            try:
                iv = int(v)
            except Exception:
                raise ValueError(f"参数 {a.name} 需要 int")
            if a.min is not None and iv < a.min:
                raise ValueError(f"参数 {a.name} 不能小于 {a.min}")
            if a.max is not None and iv > a.max:
                raise ValueError(f"参数 {a.name} 不能大于 {a.max}")
            parts += [f"--{format_flag(a.name)}", str(iv)]

        elif a.type == "float":
            try:
                fv = float(v)
            except Exception:
                raise ValueError(f"参数 {a.name} 需要 float")
            if a.min is not None and fv < float(a.min):
                raise ValueError(f"参数 {a.name} 不能小于 {a.min}")
            if a.max is not None and fv > float(a.max):
                raise ValueError(f"参数 {a.name} 不能大于 {a.max}")
            parts += [f"--{format_flag(a.name)}", str(fv)]

        elif a.type == "str":
            sv = str(v)
            parts += [f"--{format_flag(a.name)}", sv]

        elif a.type == "choice":
            if a.choices is None:
                raise ValueError(f"参数 {a.name} 缺少 choices")
            if v not in a.choices:
                raise ValueError(f"参数 {a.name} 必须是 {a.choices} 之一")
            parts += [f"--{format_flag(a.name)}", str(v)]

        elif a.type == "bool":
            bv = bool(v)
            style = (a.bool_style or "presence").lower()
            if style == "presence":
                if bv:
                    parts += [f"--{format_flag(a.name)}"]
            elif style == "negatable":
                if bv:
                    parts += [f"--{format_flag(a.name)}"]
                else:
                    parts += [f"--no-{format_flag(a.name)}"]
            else:
                raise ValueError(f"参数 {a.name} bool_style 不支持：{a.bool_style}")

        else:
            raise ValueError(f"不支持的参数类型：{a.type} ({a.name})")

    # quote string values later when assembling command
    return parts


def build_python_cmd(module: str, args_parts: List[str]) -> str:
    # Proper quoting:
    # - flags like --x remain
    # - values need quoting
    out = ["python", "-m", module]
    i = 0
    while i < len(args_parts):
        token = args_parts[i]
        if token.startswith("--"):
            out.append(token)
            # if next token is a value (not another flag) -> quote it
            if i + 1 < len(args_parts) and not args_parts[i + 1].startswith("--"):
                out.append(q(args_parts[i + 1]))
                i += 2
            else:
                i += 1
        else:
            # should not happen
            out.append(q(token))
            i += 1
    return " ".join(out)


def build_thread_env_exports(num_threads: int) -> str:
    n = int(num_threads)
    if n <= 0:
        return ""
    # If enabled, set common thread envs before running python in the tmux session
    return (
        f"export OMP_NUM_THREADS={n}\n"
        f"export MKL_NUM_THREADS={n}\n"
        f"export OPENBLAS_NUM_THREADS={n}\n"
        f"export NUMEXPR_NUM_THREADS={n}\n"
        "export OMP_PROC_BIND=true\n"
        "export OMP_PLACES=cores\n"
    )


def build_remote_script(
    *,
    workdir: str,
    conda_env: str,
    logs_dir: str,
    session: str,
    python_cmd: str,
    pre_cmds: List[str],
    thread_env_exports: str = "",
) -> Dict[str, str]:
    log_dir = logs_dir
    log_file = f"{log_dir}/{session}.log"
    safe_log_dir = quote_path_for_shell(log_dir)
    safe_log_file = quote_path_for_shell(log_file)

    pre = "\n".join([f"{c}" for c in pre_cmds]) + ("\n" if pre_cmds else "")
    thread_env = (thread_env_exports or "").rstrip("\n")
    thread_env = (thread_env + "\n") if thread_env else ""

    main_cmd = (
        f"mkdir -p {safe_log_dir}\n"
        f"cd {q(workdir)}\n"
        f"{pre}"
        f"echo '[INFO] started at: '$(date -Is)\n"
        f"echo '[INFO] workdir: {workdir}'\n"
        f"echo '[INFO] cmd: {python_cmd}'\n"
        f"source ~/.bashrc >/dev/null 2>&1 || true\n"
        f"CONDA_BASE=$(conda info --base 2>/dev/null) || true\n"
        f"[ -n \"$CONDA_BASE\" ] && [ -f \"$CONDA_BASE/etc/profile.d/conda.sh\" ] && . \"$CONDA_BASE/etc/profile.d/conda.sh\"\n"
        f"conda activate {q(conda_env)}\n"
        f"{thread_env}"
        f"{python_cmd} |& tee -a {safe_log_file}\n"
        f"echo '[INFO] finished at: '$(date -Is)\n"
    )

    # wrap in one-liner for tmux send-keys
    # Use bash -lc to run a multi-line script safely.
    one_liner = f"bash -lc {q(main_cmd)}"
    return {"log_file": log_file, "tmux_cmd": one_liner}


# -----------------------------
# SSH / tmux operations
# -----------------------------
async def ssh_run(conn: asyncssh.SSHClientConnection, cmd: str) -> asyncssh.SSHCompletedProcess:
    return await conn.run(cmd, check=False)


async def start_tmux_session(
    *,
    host: Host,
    password: str,
    session: str,
    tmux_cmd: str,
) -> None:
    async with asyncssh.connect(
        host.hostname,
        port=host.port,
        username=host.username,
        password=password,
        known_hosts=None,  # for personal tool; you can tighten later
    ) as conn:
        # Ensure tmux exists
        r = await ssh_run(conn, "command -v tmux >/dev/null 2>&1; echo $?")
        if r.stdout.strip() != "0":
            raise RuntimeError("远端未安装 tmux：请先 sudo apt-get install tmux 或 yum install tmux")

        # Check session existence
        exists = await ssh_run(conn, f"tmux has-session -t {q(session)} >/dev/null 2>&1; echo $?")
        if exists.stdout.strip() == "0":
            raise RuntimeError(f"tmux session 已存在：{session}")

        # Create session
        await ssh_run(conn, f"tmux new-session -d -s {q(session)}")

        # Send the command
        await ssh_run(conn, f"tmux send-keys -t {q(session)} {q(tmux_cmd)} C-m")


async def stream_tail(
    *,
    host: Host,
    password: str,
    log_file: str,
    n: int = 200,
):
    async with asyncssh.connect(
        host.hostname,
        port=host.port,
        username=host.username,
        password=password,
        known_hosts=None,
    ) as conn:
        safe_log_file = quote_path_for_shell(log_file)
        cmd = f"bash -lc {q(f'tail -n {n} -F {safe_log_file}')}"
        proc = await conn.create_process(cmd)

        # stream line by line
        async for line in proc.stdout:
            yield line.rstrip("\n")


def run_async(coro):
    # streamlit runs in sync context; wrap asyncio
    return asyncio.run(coro)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Remote Experiment Runner", layout="wide")
st.title("Remote Experiment Runner (Streamlit + tmux + SSH)")
st.markdown(
    """
    <style>
        div[data-testid="stDivider"] hr {margin: 0.15rem 0 !important;}
        section.main > div.block-container {padding-top: 0.5rem; padding-bottom: 0.5rem;}
        .copy-session {
            background: none;
            border: none;
            color: #1f77b4;
            cursor: pointer;
            padding: 0;
            text-decoration: underline;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = load_config()
hosts, conda_env, logs_dir, exps = parse_specs(cfg)

if "executed_commands" not in st.session_state:
	st.session_state["executed_commands"] = []
if "numactl_enabled" not in st.session_state:
	st.session_state["numactl_enabled"] = False
if "numa_node" not in st.session_state:
	st.session_state["numa_node"] = 1
if "auto_tail_trigger" not in st.session_state:
	st.session_state["auto_tail_trigger"] = False
if "tail_lines" not in st.session_state:
	st.session_state["tail_lines"] = 50

host_names = [h.name for h in hosts]
exp_names = [e.name for e in exps]

left, right = st.columns([0.45, 0.55], gap="large")

with left:
    st.subheader("1) 选择远端与实验模板")
    host_name = st.selectbox("远端机器", host_names, index=0)
    host = next(h for h in hosts if h.name == host_name)

    password = st.text_input("SSH 密码（仅本次会话内使用）", type="password")

    exp_name = st.selectbox("实验模板", exp_names, index=0)
    exp = next(e for e in exps if e.name == exp_name)

    st.caption(exp.description or "")

    st.divider()
    st.subheader("2) 参数表单")

    # init state storage key per experiment
    state_key = f"params::{exp.name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    params_state: Dict[str, Any] = st.session_state[state_key]

    # Controls for compare workflow
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset to defaults", use_container_width=True):
            params_state.clear()
    with c2:
        if st.button("Keep current (no-op)", use_container_width=True):
            pass
    with c3:
        st.write("")  # spacer

    # Render form
    with st.form(key="param_form", clear_on_submit=False):
        values: Dict[str, Any] = {}
        form_cols = st.columns(2, gap="small")

        for idx, a in enumerate(exp.args):
            col = form_cols[idx % len(form_cols)]
            with col:
                # removed toggle UI; parameters are always enabled
                cur = params_state.get(a.name, a.default)

                label = f"{a.name}"
                if a.required:
                    label += " *"
                if a.help:
                    label += f"  ({a.help})"

                if a.type == "int":
                    v = st.number_input(label, value=int(cur) if cur is not None else 0, step=1, format="%d", key=a.name)
                    values[a.name] = int(v)

                elif a.type == "float":
                    v = st.number_input(label, value=float(cur) if cur is not None else 0.0, format="%.10f", key=a.name)
                    values[a.name] = float(v)

                elif a.type == "str":
                    v = st.text_input(label, value=str(cur) if cur is not None else "", key=a.name)
                    values[a.name] = v

                elif a.type == "choice":
                    if not a.choices:
                        st.error(f"{a.name} 缺少 choices")
                        continue
                    default_index = 0
                    if cur in a.choices:
                        default_index = a.choices.index(cur)
                    v = st.selectbox(label, options=a.choices, index=default_index, key=a.name)
                    values[a.name] = v

                elif a.type == "bool":
                    v = st.checkbox(label, value=bool(cur), key=a.name)
                    values[a.name] = bool(v)

                    style = (a.bool_style or "presence").lower()
                    flag = format_flag(a.name)
                    st.caption(f"bool_style: {style}  |  True -> --{flag}" + (f", False -> --no-{flag}" if style == "negatable" else ", False -> (omit)"))

                else:
                    st.error(f"不支持的类型：{a.type}")

        submitted = st.form_submit_button("Update command preview", use_container_width=True)
        if submitted:
            # persist
            params_state.update(values)
    st.divider()
    st.subheader("已执行的命令")
    commands = st.session_state["executed_commands"]
    if commands:
        for entry in commands:
            if "id" not in entry:
                entry["id"] = f"{entry.get('session', 'sess')}_{int(dt.datetime.now().timestamp() * 1000)}"
            if "ts_raw" not in entry:
                try:
                    entry["ts_raw"] = dt.datetime.strptime(entry["ts"], "%Y-%m-%d %H:%M").timestamp()
                except Exception:
                    entry["ts_raw"] = 0.0
        for entry in sorted(commands, key=lambda e: e.get("ts_raw", 0), reverse=True):
            cols = st.columns([0.8, 0.2], gap="small")
            with cols[0]:
                copy_cmd = f"tmux a -t {entry['session']}"
                copy_payload = copy_cmd.replace("'", "\\'")
                st.markdown(
                    f"**{entry['ts']}** | Session "
                    f"<button class='copy-session' onclick=\"navigator.clipboard.writeText('{copy_payload}');return false;\">"
                    f"{entry['session']}</button>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("删除", key=f"del_{entry['id']}", use_container_width=True):
                    st.session_state["executed_commands"] = [e for e in commands if e["id"] != entry["id"]]
                    trigger_rerun()
            st.text_area(
                label=f"cmd_{entry['id']}",
                value=entry["cmd"],
                height=90,
                key=f"cmd_view_{entry['id']}",
                label_visibility="collapsed",
            )
    else:
        st.caption("暂无执行记录。")

# build the base python command so the right column can apply NUMA/session controls
active_params_state = params_state.copy()
try:
    args_parts = validate_and_build_args(exp, active_params_state)
    base_python_cmd = build_python_cmd(exp.module, args_parts)
    args_error = None
except Exception as e:
    base_python_cmd = None
    args_error = str(e)

with right:
	st.subheader("2.5) NUMA 绑定（可选）")
	use_numactl = st.checkbox(
		"在 Python 命令前加 numactl 限定",
		key="numactl_enabled",
		help="控制是否在 python 命令前追加 numactl --cpunodebind/--membind",
	)
	numa_node = st.selectbox(
		"NUMA 节点编号（0-3，cpunodebind 与 membind 共用）",
		[0, 1, 2, 3],
		key="numa_node",
		disabled=not use_numactl,
	)
	st.caption("启用后命令前会自动加：numactl --cpunodebind=<node> --membind=<node>")
	st.divider()

	st.subheader("3) Session / Run")
	session_prefix = st.text_input("session 前缀（建议写一个实验名）", value=exp.name)
	session = f"{slugify(session_prefix)}_{now_compact()}"

	python_cmd = None
	tmux_cmd = None
	log_file = None
	if base_python_cmd and not args_error:
		python_cmd = base_python_cmd
		if use_numactl:
			node = int(numa_node)
			# Keep the node exactly as selected by the user (valid options: 0..3)
			node = max(0, min(3, node))
			prefix = f"numactl --cpunodebind={node} --membind={node}"
			python_cmd = f"{prefix} {python_cmd}"

		# Enable thread env exports when num_threads > 0
		thread_env_exports = ""
		try:
			nt = int(active_params_state.get("num_threads", 0) or 0)
			if nt > 0:
				thread_env_exports = build_thread_env_exports(nt)
		except Exception:
			thread_env_exports = ""

		script_info = build_remote_script(
			workdir=host.workdir,
			conda_env=conda_env,
			logs_dir=logs_dir,
			session=session,
			python_cmd=python_cmd,
			pre_cmds=exp.pre_cmds,
			thread_env_exports=thread_env_exports,
		)
		log_file = script_info["log_file"]
		tmux_cmd = script_info["tmux_cmd"]

	if args_error:
		st.error(args_error)

	run_clicked = st.button(
		"Run on remote (tmux)",
		type="primary",
		use_container_width=True,
		disabled=(not password or not tmux_cmd),
	)

	if run_clicked:
		try:
			run_async(start_tmux_session(host=host, password=password, session=session, tmux_cmd=tmux_cmd))
			st.success(f"已启动：host={host.name} session={session}")
			st.session_state["last_run"] = {
				"host": host.name,
				"session": session,
				"log_file": log_file,
				"python_cmd": python_cmd,
				"params": active_params_state.copy(),
			}
			now_ts = dt.datetime.now()
			entry = {
				"id": f"{session}_{int(now_ts.timestamp() * 1000)}",
				"ts": now_ts.strftime("%Y-%m-%d %H:%M"),
				"ts_raw": now_ts.timestamp(),
				"session": session,
				"cmd": python_cmd,
			}
			st.session_state["executed_commands"].append(entry)
			st.session_state["auto_tail_trigger"] = True
			trigger_rerun()
		except Exception as e:
			st.error(f"启动失败：{e}")

	st.divider()
	st.subheader("命令预览（本地生成）")
	if python_cmd:
		st.text_area(
			"命令（自动换行）",
			value=python_cmd,
			height=160,
			help="命令会自动换行显示，无需进度条",
			disabled=True,
		)
		if log_file:
			st.caption(f"远端日志文件：{log_file}")
	else:
		st.info("请先生成命令后再查看预览。")
	st.divider()
	st.subheader("实时日志（tail -F）")

	last = st.session_state.get("last_run")
	if not last:
		st.info("先在左侧点击 Run 启动一个实验，然后这里会自动开始显示日志。")
	else:
		st.write(f"Host: `{last['host']}`  |  Session: `{last['session']}`")
		st.write(f"Log: `{last['log_file']}`")
		st.code(last["python_cmd"], language="bash")

		log_box = st.empty()

		# manual tail controls
		lines = st.number_input(
			"tail 最近 N 行",
			min_value=20,
			max_value=2000,
			value=int(st.session_state.get("tail_lines", 30)),
			step=20,
			key="tail_lines",
		)
		start_tail = st.button("Start/Restart tail", use_container_width=True)
		auto_tail = bool(st.session_state.get("auto_tail_trigger")) and last is not None
		if auto_tail:
			st.session_state["auto_tail_trigger"] = False
		should_tail = start_tail or auto_tail
		if should_tail:
			if not password:
				st.warning("请输入密码以 tail 日志。")
			else:
				host2 = next(h for h in hosts if h.name == last["host"])
				buf: List[str] = []
				try:
					async def _run_tail():
						async for line in stream_tail(
							host=host2,
							password=password,
							log_file=last["log_file"],
							n=int(lines),
						):
							buf.append(line)
							if len(buf) > 2000:
								del buf[:500]
							log_box.code("\n".join(buf), language="text")

					run_async(_run_tail())
				except Exception as e:
					st.error(f"tail 失败：{e}")

	st.divider()
	st.subheader("提示")
	st.write(
		"- 你也可以随时 SSH 到远端 `tmux a -t <session>` 查看。\n"
		"- 建议远端确保 `~/exp-logs` 有足够空间。\n"
		"- 如果你希望 Run 后自动开始 tail，可把右侧的按钮逻辑改成自动触发（Streamlit 的交互模型需要一点技巧）。"
	)