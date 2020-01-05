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
import com.simiacryptus.ref.lang.RefAware;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;

@SuppressWarnings("serial")
public @RefAware
class StdDevMetaLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StdDevMetaLayer.class);

  public StdDevMetaLayer() {
    this(1);
  }

  public StdDevMetaLayer(final int minBatchCount) {
    super(1);
    add(new AvgMetaLayer().setMinBatchCount(minBatchCount));
    add(new AvgReducerLayer());
    InnerNode square = add(new SqActivationLayer());
    add(new SqActivationLayer(), getInput(0), square);
    add(new AvgMetaLayer().setMinBatchCount(minBatchCount));
    add(new AvgReducerLayer());
    add(new SumInputsLayer(), getHead(), add(new LinearActivationLayer().setScale(-1).freeze(), square));
    add(new NthPowerActivationLayer().setPower(0.5));
  }

  protected StdDevMetaLayer(@Nonnull final JsonObject json,
                            Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  @SuppressWarnings("unused")
  public static StdDevMetaLayer fromJson(@NotNull final JsonObject json,
                                         Map<CharSequence, byte[]> rs) {
    return new StdDevMetaLayer(json, rs);
  }

  public static @SuppressWarnings("unused")
  StdDevMetaLayer[] addRefs(StdDevMetaLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(StdDevMetaLayer::addRef)
        .toArray((x) -> new StdDevMetaLayer[x]);
  }

  public static @SuppressWarnings("unused")
  StdDevMetaLayer[][] addRefs(StdDevMetaLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(StdDevMetaLayer::addRefs)
        .toArray((x) -> new StdDevMetaLayer[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  StdDevMetaLayer addRef() {
    return (StdDevMetaLayer) super.addRef();
  }

}
