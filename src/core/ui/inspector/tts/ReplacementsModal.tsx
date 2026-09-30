import type { TTS_State } from "@/core/services/tts/schema";
import NiceModal from "@ebay/nice-modal-react";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Modal from "../../Modal";
import { InputCheckbox, InputMapObject } from "../components/input";

// TTS word replacements dialog (`NiceModal.show('tts-replacements')`); registered when this module loads.

const WordsReplacementModal: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts);
  const up = <K extends keyof TTS_State>(key: K, v: TTS_State[K]) => window.ApiServer.patchService("tts", s => s.data[key] = v);

  return <Modal.Body width={420}>
    <Modal.Header>{t('word_replacements.title')}</Modal.Header>
    <Modal.Content>
      <div className="p-4">
        <InputCheckbox label="word_replacements.field_ignore_case" value={data.data.replaceWordsIgnoreCase} onChange={v => up("replaceWordsIgnoreCase", v)} />
        <InputMapObject keyPlaceholder={t('word_replacements.label_dictionary_key')} valuePlaceholder={t('word_replacements.label_dictionary_value')} addLabel="Add word" value={{ ...data.data.replaceWords }} onChange={e => up("replaceWords", e)} label="" />
      </div>
    </Modal.Content>
  </Modal.Body>
}
NiceModal.register('tts-replacements', (props) => <Modal.Base {...props}><WordsReplacementModal /></Modal.Base>);
