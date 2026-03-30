use serde::{Deserialize, Serialize};
use std::{os::raw::c_int, sync::RwLock};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime, State,
};
use tokio::sync::mpsc;
use windows::Win32::{
    Foundation::{LPARAM, LRESULT, WPARAM},
    System::Threading::{AttachThreadInput, GetCurrentThreadId},
    UI::{
        Input::KeyboardAndMouse::{GetKeyboardLayout, GetKeyboardState, MapVirtualKeyExW, ToUnicodeEx, MAPVK_VK_TO_VSC_EX, VK_LCONTROL},
        WindowsAndMessaging::{
            CallNextHookEx, GetForegroundWindow, GetWindowThreadProcessId, SetWindowsHookExA, UnhookWindowsHookEx, HC_ACTION, HHOOK, KBDLLHOOKSTRUCT,
            WH_KEYBOARD_LL, WM_KEYDOWN,
        },
    },
};

/// Win32 hook handles are thread-affine; we only touch them from the hook install path and Win32
/// callbacks. Marking `Send`/`Sync` matches the prior windows crate behavior and satisfies `State<T>`.
#[repr(transparent)]
struct SendHhook(Option<HHOOK>);

unsafe impl Send for SendHhook {}
unsafe impl Sync for SendHhook {}

#[allow(dead_code)]
struct BgInput {
    tx: mpsc::UnboundedSender<String>,
    listen_hook_id: RwLock<SendHhook>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum KeyCommand {
    Escape,
    Return,
    Delete,
    BackSpace,
    Key(String),
}

#[allow(dead_code, static_mut_refs)]
static mut GLOBAL_CALLBACK: Option<Box<dyn FnMut(KeyCommand)>> = None;

#[allow(dead_code)]
unsafe extern "system" fn raw_callback(code: c_int, param: WPARAM, lpdata: LPARAM) -> LRESULT {
    if code as u32 != HC_ACTION {
        return CallNextHookEx(None, code, param, lpdata);
    }

    if let Ok(WM_KEYDOWN) = param.0.try_into() {
        let KBDLLHOOKSTRUCT { vkCode, .. } = *(lpdata.0 as *const KBDLLHOOKSTRUCT);
        let m: Option<KeyCommand> = match vkCode {
            46 => Some(KeyCommand::Delete),
            27 => Some(KeyCommand::Escape),
            8 => Some(KeyCommand::BackSpace),
            13 => Some(KeyCommand::Return),
            _ => {
                let window_thread_id = GetWindowThreadProcessId(GetForegroundWindow(), None);
                let thread_id = GetCurrentThreadId();

                let mut kb_state = [0_u8; 256_usize];
                if AttachThreadInput(thread_id, window_thread_id, true).as_bool() {
                    let _ = GetKeyboardState(&mut kb_state);
                    let _ = AttachThreadInput(thread_id, window_thread_id, false);
                } else {
                    let _ = GetKeyboardState(&mut kb_state);
                }

                if kb_state[VK_LCONTROL.0 as usize] > 1 {
                    None
                } else {
                    let kb_layout = GetKeyboardLayout(window_thread_id);
                    let code = MapVirtualKeyExW(vkCode, MAPVK_VK_TO_VSC_EX, Some(kb_layout)) << 16;

                    let mut name = [0_u16; 32];
                    let res_size = ToUnicodeEx(vkCode, code, &kb_state, &mut name, 0, Some(kb_layout));
                    if res_size > 0 {
                        if let Some(s) = String::from_utf16(&name[..res_size as usize]).ok() {
                            Some(KeyCommand::Key(s))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            }
        };

        if m.is_some() {
            #[allow(static_mut_refs)]
            if let Some(cb) = &mut GLOBAL_CALLBACK {
                if let Some(cmd) = m {
                    cb(cmd);
                }
            }
            // block on command
            return LRESULT(1);
        }
    }
    CallNextHookEx(None, code, param, lpdata)
}

#[allow(dead_code)]
#[command]
fn start_tracking(state: State<'_, BgInput>) -> Result<(), String> {
    {
        let current_hook_id = state
            .listen_hook_id
            .read()
            .map_err(|_| "Failed to read lock")?;
        if current_hook_id.0.is_some() {
            return Err("Already active".into());
        }
    }

    let tx = state.tx.clone();
    unsafe {
        GLOBAL_CALLBACK = Some(Box::new(move |cmd| {
            let rpc: String = match cmd {
                KeyCommand::Escape => "cmd:cancel".to_string(),
                KeyCommand::Return => "cmd:submit".to_string(),
                KeyCommand::Delete | KeyCommand::BackSpace => "cmd:delete".to_string(),
                KeyCommand::Key(key) => format!("key:{}", key),
            };
            tx.send(rpc).ok();
        }));
    }
    let hook = unsafe { SetWindowsHookExA(WH_KEYBOARD_LL, Some(raw_callback), None, 0) }
        .map_err(|_| "Could not start listener".to_string())?;
    let mut wr = state
        .listen_hook_id
        .write()
        .map_err(|_| "Failed to write lock")?;
    wr.0 = Some(hook);
    Ok(())
}

#[allow(dead_code)]
#[command]
fn stop_tracking(state: State<'_, BgInput>) {
    if let Ok(mut wr) = state.listen_hook_id.write() {
        if let Some(hook) = wr.0.take() {
            unsafe {
                let _ = UnhookWindowsHookEx(hook);
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
struct HotkeyBinding {
    name: String,
}

#[cfg(feature = "background_input")]
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    let (pubsub_output_tx, mut pubsub_output_rx) = mpsc::unbounded_channel::<String>(); // to js
    Builder::new("keyboard")
        .invoke_handler(tauri::generate_handler![start_tracking, stop_tracking])
        .setup(|app, _api| {
            app.manage(BgInput {
                tx: pubsub_output_tx,
                listen_hook_id: RwLock::new(SendHhook(None)),
            });
            let handle = app.clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    if let Some(output) = pubsub_output_rx.recv().await {
                        let _ = handle.emit("keyboard", output);
                    }
                }
            });
            Ok(())
        })
        .build()
}

#[cfg(not(feature = "background_input"))]
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("keyboard").build()
}
