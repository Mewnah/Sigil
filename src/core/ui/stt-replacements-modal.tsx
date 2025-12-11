import { FC } from "react";
import { useTranslation } from "react-i18next";
import NiceModal from "@ebay/nice-modal-react";
import { useSnapshot } from "valtio";
import { STT_State } from "@/core/services/stt/schema";
import Modal from "./Modal";
import Inspector from "./inspector/components";
import { InputCheckbox, InputMapObject } from "./inspector/components/input";

const WordsReplacementModal: FC = () => {
    const { t } = useTranslation();
    const data = useSnapshot(window.ApiServer.state.services.stt);

    const up = <K extends keyof STT_State>(key: K, v: STT_State[K]) => window.ApiServer.patchService("stt", s => s.data[key] = v);

    return <Modal.Body width={420}>
        <Modal.Header>{t('word_replacements.title')}</Modal.Header>
        <Modal.Content>
            <div className="p-4 flex flex-col space-y-2">
                <InputCheckbox label="word_replacements.field_ignore_case" value={data.data.replaceWordsIgnoreCase} onChange={v => up("replaceWordsIgnoreCase", v)} />
                {data.data.replaceWordsIgnoreCase && <>
                    <InputCheckbox label="word_replacements.field_preserve_capitalization" value={data.data.replaceWordsPreserveCase} onChange={v => up("replaceWordsPreserveCase", v)} />
                    <Inspector.Description>{t('word_replacements.field_preserve_capitalization_desc')}</Inspector.Description>
                </>}
                <InputMapObject keyPlaceholder={t('word_replacements.label_dictionary_key')} valuePlaceholder={t('word_replacements.label_dictionary_value')} addLabel={t('common.btn_add')} value={{ ...data.data.replaceWords }} onChange={e => up("replaceWords", e)} label="" />
            </div>
        </Modal.Content>
    </Modal.Body>
}

NiceModal.register('stt-replacements', (props) => <Modal.Base {...props}><WordsReplacementModal /></Modal.Base>);
