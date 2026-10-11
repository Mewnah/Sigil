# Close-AppWindow: ask the window of ONE process that has exactly this title to close (WM_CLOSE), like the close button.
#
# Dot-source it, then:  Close-AppWindow -ProcessId $app.Id -Title 'Sigil'   (returns how many windows were asked)
#
# Why not Process.CloseMainWindow(): Sigil's event loop (tao) owns an invisible tool window (class "Tao Thread Event Target",
# no title, 16 x 16) that Windows counts as a visible, owner-less top-level window, and once the real window is minimized
# (SIGIL_START_MINIMIZED, every automated launch) it is the one CloseMainWindow() picks. WM_CLOSE there destroys the event
# loop's own message window: the real window closes, but the process lingers until the app's exit guard ends it 20 s later
# (src-tauri/src/exit_guard.rs, scripts/release/README.md). The title match plus the process id keeps this from touching
# another Sigil that the user has open.
if (-not ('FootprintWin.Close' -as [type])) {
  Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
namespace FootprintWin {
  public static class Close {
    delegate bool EnumProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc p, IntPtr l);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] static extern bool PostMessage(IntPtr h, uint msg, IntPtr w, IntPtr l);
    public static int Titled(int pid, string title) {
      int n = 0;
      EnumWindows((h, l) => {
        uint p; GetWindowThreadProcessId(h, out p);
        if (p != (uint)pid) return true;
        StringBuilder t = new StringBuilder(256); GetWindowText(h, t, 256);
        if (t.ToString() == title && PostMessage(h, 0x0010, IntPtr.Zero, IntPtr.Zero)) n++;
        return true;
      }, IntPtr.Zero);
      return n;
    }
  }
}
'@
}

function Close-AppWindow {
  param([Parameter(Mandatory)][int]$ProcessId, [string]$Title = 'Sigil')
  return [FootprintWin.Close]::Titled($ProcessId, $Title)
}
