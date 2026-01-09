import { Kick_State } from "@/core/services/kick/schema";
import { ServiceNetworkState } from "@/types";
import { FC } from "react";
import { useSnapshot, proxy } from "valtio";
import Inspector from "./components";
import { InputCheckbox, InputMapObject, InputNetworkStatus, InputText, InputTextSource } from "./components/input";
import NiceModal from "@ebay/nice-modal-react";
import Modal from "../Modal";
import { useTranslation } from "react-i18next";

// Kick logo as inline SVG component
const KickLogo: FC<{ size?: number, className?: string }> = ({ size = 20, className }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="currentColor"
        className={className}
    >
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
    </svg>
);

// Simpler Kick "K" logo
const KickIcon: FC<{ size?: number }> = ({ size = 20 }) => (
    <svg width={size} height={size} viewBox="0 0 512 512" fill="currentColor">
        <path d="M115.2 0h76.8v204.8L345.6 0H448L268.8 230.4 460.8 512H345.6L192 281.6V512h-76.8V0z" />
    </svg>
);

const KickEmotesModal: FC = () => {
    return (
        <Modal.Body width={420}>
            <Modal.Header>Kick Emotes</Modal.Header>
            <Modal.Content>
                <div className="p-4 flex flex-col space-y-2">
                    <p className="text-sm opacity-50">Emote inspection not yet implemented for Kick.</p>
                </div>
            </Modal.Content>
        </Modal.Body>
    );
}
NiceModal.register('kick-emotes', (props) => <Modal.Base {...props}><KickEmotesModal /></Modal.Base>);

const KickEmotesMapModal: FC = () => {
    const { t } = useTranslation();
    const kickData = window.ApiServer.state.services.kick?.data;
    if (!kickData) return null;
    const pr = useSnapshot(kickData);
    const up = <K extends keyof Kick_State>(key: K, v: Kick_State[K]) => window.ApiServer.patchService("kick", s => s.data[key] = v);
    return (
        <Modal.Body width={420}>
            <Modal.Header>{t('twitch_emotes_remap.title')}</Modal.Header>
            <Modal.Content>
                <div className="p-4">
                    <InputMapObject keyPlaceholder={t('twitch_emotes_remap.label_dictionary_key')} valuePlaceholder={t('twitch_emotes_remap.label_dictionary_value')} value={{ ...pr.emotesReplacements }} onChange={e => up("emotesReplacements", e)} label="common.field_dictionary" />
                </div>
            </Modal.Content>
        </Modal.Body>
    );
}
NiceModal.register('kick-emotes-map', (props) => <Modal.Base {...props}><KickEmotesMapModal /></Modal.Base>);

const Inspector_Kick: FC = () => {
    const { t } = useTranslation();

    // Safe access with fallback
    const kickService = window.ApiServer.kick;
    const kickState = kickService?.state || proxy({ user: null, liveStatus: ServiceNetworkState.disconnected });
    const { user, liveStatus } = useSnapshot(kickState);

    const handleLogin = () => kickService?.login();
    const handleLogout = () => kickService?.logout();

    const handleShowEmotes = () => NiceModal.show('kick-emotes');
    const handleShowEmotesMapper = () => NiceModal.show('kick-emotes-map');

    // Safe access with fallback for service data
    const kickData = window.ApiServer.state.services.kick?.data;
    if (!kickData) {
        return <Inspector.Body>
            <Inspector.Header><KickIcon /> Kick Integration</Inspector.Header>
            <Inspector.Content>
                <div className="p-4 text-warning">Kick service not initialized. Please reload the application.</div>
            </Inspector.Content>
        </Inspector.Body>;
    }

    const pr = useSnapshot(kickData);
    const up = <K extends keyof Kick_State>(key: K, v: Kick_State[K]) => window.ApiServer.patchService("kick", s => s.data[key] = v);

    return <Inspector.Body>
        <Inspector.Header><KickIcon /> Kick Integration</Inspector.Header>
        <Inspector.Content>
            <Inspector.SubHeader>Kick Configuration</Inspector.SubHeader>
            {user && <div className="flex items-center space-x-4">
                {user.profilePictureUrl && <img className="rounded-full aspect-square w-10 ring-2 ring-success ring-offset-base-100 ring-offset-2" src={user.profilePictureUrl} alt={user.name} />}
                {!user.profilePictureUrl && <div className="rounded-full aspect-square w-10 bg-[#53fc18] flex items-center justify-center font-bold text-lg text-black">{user.name[0].toUpperCase()}</div>}
                <div className="flex flex-col">
                    <div className="font-semibold">{user.name}</div>
                    <div className="text-xs link link-warning link-hover font-medium" onClick={handleLogout}>{t('twitch.btn_logout')}</div>
                </div>
            </div>}

            {!user && <button className="btn gap-2 border-none bg-[#53fc18] text-black hover:bg-[#42ca13]" onClick={handleLogin}><KickIcon size={20} /> Login with Kick</button>}

            <Inspector.Switchable visible={!!user}>
                <InputNetworkStatus label="twitch.status_stream" value={liveStatus} />

                <InputCheckbox label="twitch.field_enable_chat" value={pr.chatEnable} onChange={e => up("chatEnable", e)} />
                <Inspector.Switchable visible={pr.chatEnable}>
                    <InputCheckbox label="twitch.field_post_in_chat" value={pr.chatPostEnable} onChange={e => up("chatPostEnable", e)} />
                    <Inspector.Description>{t('twitch.field_post_in_chat_desc')}</Inspector.Description>
                    <InputCheckbox label="twitch.field_post_in_chat_live" value={pr.chatPostLive} onChange={e => up("chatPostLive", e)} />
                    <InputText label="twitch.field_post_in_chat_delay" type="number" value={pr.chatSendDelay} onChange={e => up("chatSendDelay", e.target.value)} />
                    <InputTextSource label="common.field_text_source" value={pr.chatPostSource} onChange={e => up("chatPostSource", e)} />
                    <InputCheckbox label="common.field_use_keyboard_input" value={pr.chatPostInput} onChange={e => up("chatPostInput", e)} />
                    <InputCheckbox label="twitch.field_chat_text" value={pr.chatReceiveEnable} onChange={e => up("chatReceiveEnable", e)} />
                    <Inspector.Description>{t('twitch.field_chat_text_desc')}</Inspector.Description>
                </Inspector.Switchable>

                <Inspector.SubHeader>{t('twitch.section_emotes')}</Inspector.SubHeader>
                <span>
                    <span className="link link-accent link-hover font-semibold text-xs" onClick={handleShowEmotes}>{t('twitch.btn_show_emotes')}</span> | <span className="link link-accent link-hover font-semibold text-xs" onClick={handleShowEmotesMapper}>{t('twitch.btn_remap_emotes')}</span>
                </span>
                <div className="contents">
                    <InputCheckbox label="twitch.field_enable_captions_emotes" value={pr.emotesEnableReplacements} onChange={e => up("emotesEnableReplacements", e)} />
                    <InputCheckbox label="twitch.field_case_sensitive" value={pr.emotesCaseSensitive} onChange={e => up("emotesCaseSensitive", e)} />
                </div>
            </Inspector.Switchable>
        </Inspector.Content>
    </Inspector.Body>
}

export default Inspector_Kick;
