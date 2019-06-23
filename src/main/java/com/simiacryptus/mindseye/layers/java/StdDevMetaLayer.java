/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;

@SuppressWarnings("serial")
public class StdDevMetaLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StdDevMetaLayer.class);

  public StdDevMetaLayer() {
    this(1);
  }

  public StdDevMetaLayer(final int minBatchCount) {
    super(1);
    wrap(new AvgMetaLayer().setMinBatchCount(minBatchCount)).freeRef();
    wrap(new AvgReducerLayer()).freeRef();
    InnerNode square = wrap(new SqActivationLayer());
    wrap(new SqActivationLayer(), getInput(0), square.addRef()).freeRef();
    wrap(new AvgMetaLayer().setMinBatchCount(minBatchCount)).freeRef();
    wrap(new AvgReducerLayer()).freeRef();
    wrap(new SumInputsLayer(), getHead(), wrap(new LinearActivationLayer().setScale(-1).freeze(), square)).freeRef();
    wrap(new NthPowerActivationLayer().setPower(0.5)).freeRef();
  }

  protected StdDevMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  public static StdDevMetaLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StdDevMetaLayer(json, rs);
  }

}
